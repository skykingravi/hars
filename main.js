// Import TensorFlow.js
import * as tf from "@tensorflow/tfjs";

const LABELS = ["Downstairs", "Jogging", "Sitting", "Upstairs", "Walking"];
const argmax = (array) => {
    let maxi = -Infinity,
        indx;
    for (let i = 0; i < array.length; i++) {
        if (maxi < array[i]) {
            maxi = array[i];
            indx = i;
        }
    }
    return indx;
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

    // Array to store accelerometer data
    let accelerometerData = [];

    window.addEventListener("devicemotion", handleDeviceMotion, {
        frequency: 50,
    });

    function handleDeviceMotion(event) {
        // Extract acceleration data from the event
        const acceleration =
            event.accelerationIncludingGravity || event.acceleration;

        // Check if the acceleration data is available
        if (acceleration) {
            const dataPoint = [acceleration.x, acceleration.y, acceleration.z];
            accelerometerData.push(dataPoint);

            // When we have 100 data points, make a prediction and reset the array
            if (accelerometerData.length === 100) {
                makePredictionAndReset();
            }
        } else {
            document.body.innerHTML = "<h1>Not Supported</h1>";
        }
    }

    function makePredictionAndReset() {
        try {
            // Convert the 2D array (100, 3) to a 3D tensor (1, 100, 3)
            const inputTensor = tf.tensor3d([accelerometerData]);

            // Normalize the input data using Min-Max scaling
            const min = tf.min(inputTensor, [1], true);
            const max = tf.max(inputTensor, [1], true);

            const normalizedInput = tf.div(
                tf.sub(inputTensor, min),
                tf.sub(max, min)
            );

            // Make predictions
            const predictions = model.predict(normalizedInput);

            // Display predictions
            predictions.print();
            document.body.innerHTML = `<h1>${LABELS[argmax(predictions)]}</h1>`;

            // Reset the accelerometer data array
            accelerometerData = [];
        } catch (e) {
            alert(e);
        }
    }
});
