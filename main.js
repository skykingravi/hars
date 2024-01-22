// Import TensorFlow.js
import * as tf from "@tensorflow/tfjs";

const LABELS = [
    "biking",
    "downstairs",
    "jogging",
    "sitting",
    "standing",
    "upstairs",
    "walking",
];
const argmax = (array) => {
    return array.indexOf(Math.max(...array));
};

document.addEventListener("DOMContentLoaded", async function () {
    // Request accelerometer permission
    if (
        typeof DeviceMotionEvent !== "undefined" &&
        typeof DeviceMotionEvent.requestPermission === "function"
    ) {
        try {
            await DeviceMotionEvent.requestPermission();
            console.log("Accelerometer permission granted.");
        } catch (error) {
            console.error("Failed to request accelerometer permission:", error);
            return;
        }
    }
    // Load the TensorFlow.js model
    const model = await tf.loadLayersModel("./model.json");
    model.summary();

    // Initialize arrays to store sensor data
    let linearAccelerationData = [];
    let gravityData = [];
    let rotationRateData = [];

    // Function to add sensor data
    function addSensorData(linearAcc, gravity, rotationRate) {
        // Push data to the respective arrays
        linearAccelerationData.push(linearAcc);
        gravityData.push(gravity);
        rotationRateData.push(rotationRate);

        // If we have 100 records, make predictions and reset the arrays
        if (
            linearAccelerationData.length === 100 &&
            gravityData.length === 100 &&
            rotationRateData.length === 100
        ) {
            makePredictionsAndReset();
        }
    }

    // Function to make predictions and reset the arrays
    function makePredictionsAndReset() {
        try {
            // Convert sensor data arrays to 3D tensors
            const linearAccTensor = tf.tensor2d(linearAccelerationData);
            const gravityTensor = tf.tensor2d(gravityData);
            const rotationRateTensor = tf.tensor2d(rotationRateData);

            // Concatenate the tensors along the last axis (axis=1) to get a single input tensor
            const inputTensor = tf.concat(
                [gravityTensor, linearAccTensor, rotationRateTensor],
                1
            );

            // Make predictions
            const predictions = model.predict(inputTensor);

            // Convert predictions tensor to JavaScript array
            const predictionsArray = predictions.arraySync();

            document.body.innerHTML += `<h1>${
                LABELS[argmax(predictionsArray[0])]
            }</h1>`;

            // Find the index of the predicted activity
            const maxIndex = predictionsArray[0].indexOf(
                Math.max(...predictionsArray[0])
            );

            // Log the predicted activity index (adjust based on your activity classes)
            console.log("Predicted Activity Index:", maxIndex);

            // Reset the sensor data arrays
            linearAccelerationData = [];
            gravityData = [];
            rotationRateData = [];
        } catch (e) {
            alert(e);
        }
    }

    // Add event listeners for sensor data
    window.addEventListener("devicemotion", handleDeviceMotion, {
        frequency: 12,
    });

    function handleDeviceMotion(event) {
        // Extract relevant sensor data from the event
        const linearAcceleration = event.acceleration;
        const gravity = event.accelerationIncludingGravity;
        const rotationRate = event.rotationRate;

        // Check if the sensor data is available
        if (linearAcceleration && gravity && rotationRate) {
            const linearAccDataPoint = [
                linearAcceleration.x,
                linearAcceleration.y,
                linearAcceleration.z,
            ];
            const gravityDataPoint = [gravity.x, gravity.y, gravity.z];
            const rotationRateDataPoint = [
                rotationRate.alpha,
                rotationRate.beta,
                rotationRate.gamma,
            ];

            // Add sensor data to the arrays
            addSensorData(
                linearAccDataPoint,
                gravityDataPoint,
                rotationRateDataPoint
            );
        }
    }
});
