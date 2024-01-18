// Import TensorFlow.js
import * as tf from "@tensorflow/tfjs";

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
    const model = await tf.loadLayersModel("./assets/model.json");

    // Array to store accelerometer data
    let accelerometerData = [];

    window.addEventListener("devicemotion", handleDeviceMotion, {
        frequency: 50,
    });

    function handleDeviceMotion(event) {
        // Extract acceleration data from the event
        const acceleration =
            event.accelerationIncludingGravity || event.acceleration;

        document.body.innerHTML = `<h1>${acceleration}</h1>`;

        // Check if the acceleration data is available
        if (acceleration) {
            const dataPoint = [acceleration.x, acceleration.y, acceleration.z];
            accelerometerData.push(dataPoint);

            // When we have 100 data points, make a prediction and reset the array
            if (accelerometerData.length === 100) {
                makePredictionAndReset();
            }
        } else {
            document.body = "<h1>Not Supported</h1>";
        }
    }

    function makePredictionAndReset() {
        alert("HI");
        // Convert the 2D array (100, 3) to a 3D tensor (1, 100, 3)
        const inputTensor = tf.tensor3d([accelerometerData]);

        // Normalize the input data using Min-Max scaling
        const min = tf.min(inputTensor, (axis = 1), (keepDims = true));
        const max = tf.max(inputTensor, (axis = 1), (keepDims = true));

        const normalizedInput = tf.div(
            tf.sub(inputTensor, min),
            tf.sub(max, min)
        );

        // Make predictions
        const predictions = model.predict(normalizedInput);

        // Display predictions
        predictions.print();
        document.body = `<h1>${predictions}</h1>`;

        // Reset the accelerometer data array
        accelerometerData = [];
    }
});
