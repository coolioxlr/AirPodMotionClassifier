
import UIKit
import SceneKit
import CoreMotion
import CoreML
import simd

extension float4x4 {
    init(rotationMatrix r: CMRotationMatrix) {
        self.init([
            simd_float4(Float(-r.m11), Float(r.m13), Float(r.m12), 0.0),
            simd_float4(Float(-r.m31), Float(r.m33), Float(r.m32), 0.0),
            simd_float4(Float(-r.m21), Float(r.m23), Float(r.m22), 0.0),
            simd_float4(          0.0,          0.0,          0.0, 1.0)
        ])
    }
}

class HeadphonePoseViewController: UIViewController, CMHeadphoneMotionManagerDelegate {
    
    @IBOutlet weak var sceneView: SCNView!
    @IBOutlet weak var motionButton: UIButton!
    @IBOutlet weak var referenceButton: UIButton!
    
    @IBOutlet weak var motionLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    private var motionManager = CMHeadphoneMotionManager()
    private var headNode: SCNNode?
    private var referenceFrame = matrix_identity_float4x4
    
    // Define some ML Model constants for the recurrent network
    struct ModelConstants {
        static let numOfFeatures = 6
        // Must be the same value you used while training
        static let predictionWindowSize = 20
        // Must be the same value you used while training
        static let sensorsUpdateFrequency = 1.0 / 10.0
        static let hiddenInLength = 200
        static let hiddenCellInLength = 200
        static let stateInLength = 400
    }
    
    // Initialize the model, layers, and prediction window
    private let classifier = HeadClassifier()
    private let modelName:String = "HeadClassifier"
    var currentIndexInPredictionWindow = 0
    //    let predictionWindowDataArray = try? MLMultiArray(shape: [1, ModelConstants.predictionWindowSize, ModelConstants.numOfFeatures] as [NSNumber], dataType: MLMultiArrayDataType.double)
    //    var lastHiddenOutput = try? MLMultiArray(shape: [ModelConstants.hiddenInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    //    var lastHiddenCellOutput = try? MLMultiArray(shape: [ModelConstants.hiddenCellInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    
    let accX = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let accY = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let accZ = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    
    let rotX = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let rotY = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let rotZ = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    
    var stateOutput = try? MLMultiArray(shape:[ModelConstants.stateInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.bringSubviewToFront(referenceButton)
        view.bringSubviewToFront(motionButton)
        
        let scene = SCNScene(named: "head.obj")!
        
        headNode = scene.rootNode.childNodes.first
        
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        scene.rootNode.addChildNode(cameraNode)
        
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 2.0)
        cameraNode.camera?.zNear = 0.05
        
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light!.type = .omni
        lightNode.position = SCNVector3(x: 0, y: 10, z: 10)
        scene.rootNode.addChildNode(lightNode)
        
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light!.type = .ambient
        ambientLightNode.light!.color = UIColor.darkGray
        scene.rootNode.addChildNode(ambientLightNode)
        
        sceneView.scene = scene
        
        motionManager.delegate = self
        
        updateButtonState()
    }
    
    private func updateButtonState() {
        motionButton.isEnabled = motionManager.isDeviceMotionAvailable
            && CMHeadphoneMotionManager.authorizationStatus() != .denied
        let motionTitle = motionManager.isDeviceMotionActive ? "Stop Tracking" : "Start Tracking"
        motionButton.setTitle(motionTitle, for: [.normal])
        referenceButton.isHidden = !motionManager.isDeviceMotionActive
    }
    
    private func toggleTracking() {
        switch CMHeadphoneMotionManager.authorizationStatus() {
        case .authorized:
            print("User previously allowed motion tracking")
        case .restricted:
            print("User access to motion updates is restricted")
        case .denied:
            print("User denied access to motion updates; will not start motion tracking")
            return
        case .notDetermined:
            print("Permission for device motion tracking unknown; will prompt for access")
        default:
            break
        }
        
        if !motionManager.isDeviceMotionActive {
            weak var weakSelf = self
            motionManager.startDeviceMotionUpdates(to: OperationQueue.main) { (maybeDeviceMotion, maybeError) in
                if let strongSelf = weakSelf {
                    if let deviceMotion = maybeDeviceMotion {
                        strongSelf.headphoneMotionManager(strongSelf.motionManager, didUpdate:deviceMotion)
                    } else if let error = maybeError {
                        strongSelf.headphoneMotionManager(strongSelf.motionManager, didFail:error)
                    }
                }
            }
            print("Started device motion updates")
        } else {
            motionManager.stopDeviceMotionUpdates()
            print("Stop device motion updates")
        }
        updateButtonState()
    }
    
    // MARK: - UIViewController
    
    override var shouldAutorotate: Bool {
        return true
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        if UIDevice.current.userInterfaceIdiom == .phone {
            return .allButUpsideDown
        } else {
            return .all
        }
    }
    
    // MARK: - IBActions
    
    @IBAction func startMotionTrackingButtonTapped(_ sender: UIButton)
    {
        toggleTracking()
    }
    
    @IBAction func referenceFrameButtonWasTapped(_ sender: UIButton)
    {
        if let deviceMotion = motionManager.deviceMotion {
            referenceFrame = float4x4(rotationMatrix: deviceMotion.attitude.rotationMatrix).inverse
        }
    }
    
    // MARK: - CMHeadphoneMotionManagerDelegate
    
    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        print("Headphones did connect")
        updateButtonState()
    }
    
    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        print("Headphones did disconnect")
        updateButtonState()
    }
    
    // MARK: Headphone Device Motion Handlers
    
    func headphoneMotionManager(_ motionManager: CMHeadphoneMotionManager, didUpdate deviceMotion: CMDeviceMotion) {
        
        addMotionDataSampleToArray(motionSample: deviceMotion)
        
        let rotation = float4x4(rotationMatrix: deviceMotion.attitude.rotationMatrix)
        
        let mirrorTransform = simd_float4x4([
            simd_float4(-1.0, 0.0, 0.0, 0.0),
            simd_float4( 0.0, 1.0, 0.0, 0.0),
            simd_float4( 0.0, 0.0, 1.0, 0.0),
            simd_float4( 0.0, 0.0, 0.0, 1.0)
        ])
        
        headNode?.simdTransform = mirrorTransform * rotation * referenceFrame
        
        updateButtonState()
    }
    
    func headphoneMotionManager(_ motionManager: CMHeadphoneMotionManager, didFail error: Error) {
        updateButtonState()
    }
    
    //Mark - Activity Classifier
    
    //---Aggregating sensor readings---
    //Whenever a new reading has been received from the sensor,
    //we will add it to our prediction_window long data array.
    //When the array is full, the application is ready to call
    //the model and get a new activity prediction.
    func addMotionDataSampleToArray(motionSample: CMDeviceMotion) {
        // Add the current motion data reading to the data array
        // Using global queue for building prediction array
        self.rotX[self.currentIndexInPredictionWindow] = motionSample.rotationRate.x as NSNumber
        self.rotY[self.currentIndexInPredictionWindow] = motionSample.rotationRate.y as NSNumber
        self.rotZ[self.currentIndexInPredictionWindow] = motionSample.rotationRate.z as NSNumber
        self.accX[self.currentIndexInPredictionWindow] = motionSample.userAcceleration.x as NSNumber
        self.accY[self.currentIndexInPredictionWindow] = motionSample.userAcceleration.y as NSNumber
        self.accZ[self.currentIndexInPredictionWindow] = motionSample.userAcceleration.z as NSNumber
        
        // Update prediction array index
        self.currentIndexInPredictionWindow += 1
        
        // If data array is full - execute a prediction
        if (self.currentIndexInPredictionWindow == ModelConstants.predictionWindowSize) {
            //DispatchQueue.main.async {
            self.activityPrediction()
            //}
            
            // Start a new prediction window from scratch
            self.currentIndexInPredictionWindow = 0
        }
    }
    
    //Making prediction
    //After prediction_window readings are aggregated,
    //we call the model to get a prediction of the current user's activity.
    func activityPrediction(){
        // Perform prediction
        let modelPrediction = try? classifier.prediction(UA_X: accX, UA_Y: accY, UA_Z: accZ, X: rotX, Y: rotY, Z: rotZ, stateIn: stateOutput)
        // Update the state vector
        stateOutput = modelPrediction?.stateOut
        
        DispatchQueue.main.async {
            if let activity = modelPrediction?.label, let probability = modelPrediction?.labelProbability[activity]{
                self.motionLabel.text = activity
                self.confidenceLabel.text = String(format: "%.0f", probability * 100) + "%"
            }
        }
    }
}
