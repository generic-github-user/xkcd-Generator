// Main JavaScript for xkcd Generator

// Define settings
const numParameters = 3;
const numLayers = 4;
// Size of input and output images in pixels (width and height)
const imageSize = 4;
// Number of images to use when training the neural network
const numTrainingImages = 15;
const logData = true;

// Automatically generated settings and parameters
// Volume of image data, calculated by squaring imageSize to find the area of the image (total number of pixels) and multiplying by three for each color channel (RGB)
const imageVolume = (imageSize ** 2) * 3;
// Get information for canvas
const canvas = document.getElementById("canvas");
// Get context for canvas
const context = canvas.getContext("2d");
const parameters = {
	"training": tf.randomNormal([15, numParameters]),
	"display": tf.randomNormal([1, numParameters])
}
var iteration = 0;

// Set canvas dimensions to match specified image dimensions
// Input canvas
canvas.width = imageSize;
canvas.height = imageSize;

// Define generator network with the high-level TensorFlow.js layers system
// This network takes a low-dimensional input image and reduces it to a low-dimensional "latent-space" representation
// Define encoder network layers
if (logData) {
	console.log("Generative adversarial network layer sizes");
}

const generator = {
	"model": tf.sequential(),
	"calculateLoss": () => tf.tidy(
		// Calculate loss
		// Do we need these tidys?
		() => {
			// Evaluate the loss function given the output of the autoencoder network and the actual image
			return loss(
				discriminator.model.predict(
					generator.model.predict(parameters.training)
				),
				tf.ones([15, numParameters])
			);
		}
	),
	"optimizer": tf.train.adam(0.0001)
};

if (logData) {
	console.log("Generator");
	console.log(numParameters);
}
generator.model.add(tf.layers.dense({units: numParameters, inputShape: [numParameters]}));
for (var i = 0; i < numLayers; i ++) {
	const layerSize = Math.round(imageVolume / (2 ** ((numLayers - 1) - i)));
	generator.model.add(tf.layers.dense({units: layerSize, activation: "sigmoid"}));
	if (logData) {
		console.log(layerSize);
	}
}

// Define generator network with the high-level TensorFlow.js layers system
// This network takes a low-dimensional input image and reduces it to a low-dimensional "latent-space" representation
// Define encoder network layers
const discriminator = {
	"model": tf.sequential(),
	"calculateLoss": () => tf.tidy(
		// Calculate loss
		() => {
			// Evaluate the loss function given the output of the autoencoder network and the actual image
			return loss(
				discriminator.model.predict(trainingData.tensor.input),
				trainingData.tensor.output
			);
		}
	),
	"optimizer": tf.train.adam(0.001)
};

if (logData) {
	console.log("Discriminator");
	console.log(imageVolume);
}
discriminator.model.add(tf.layers.dense({units: imageVolume, inputShape: [imageVolume]}));
for (var i = 0; i < numLayers; i ++) {
	const layerSize = Math.round(imageVolume / (2 ** (i + 1)));
	discriminator.model.add(tf.layers.dense({units: layerSize, activation: "sigmoid"}));
	if (logData) {
		console.log(layerSize);
	}
}

// Neural network training/optimization
// Define loss function for neural network training: Mean squared error
loss = (input, output) => input.sub(output).square().mean();

// Create object to store training data in image, pixel, and tensor format
const trainingData = {
	// Store training data as HTML image elements
	"images": [],
	// Store training data as raw arrays of pixel data
	"pixels": {
		input: [],
		output: []
	},
	// Store training data as a TensorFlow.js tensor
	"tensor": {}
}

// Add training data to trainingData.images array as an HTML image element
// Loop through each training image

for (var i = 0; i < numTrainingImages; i ++) {
	// Create a new HTML image element with the specified dimensions and set current array index to this element (array.push does not work here)
	trainingData.images[i] = new Image(imageSize, imageSize);
}

// Wait for last image (testing data) to load before continuing
trainingData.images[trainingData.images.length - 1].onload = function () {
	var pixels;
	var pixelsArray;
	var outputValues;
	function generateTrainingData() {
		trainingData.pixels.input = [];
		trainingData.pixels.output = [];
		if (trainingData.tensor.input) {
			trainingData.tensor.input.dispose();
		}
		if (trainingData.tensor.output) {
			trainingData.tensor.output.dispose();
		}
		// Create training data from pixels of image elements
		// Create a new variable to store the data
		// Loop through each training image

		for (var i = 0; i < numTrainingImages; i ++) {
			// Create a tensor with 3 (RGB) color channels from the image element
			pixels =
			tf.tidy(
				() => {
					// Resize image to the specified dimensions with resizeBilinear()
					return tf.image.resizeBilinear(
						tf.fromPixels(trainingData.images[i], numParameters),
						[imageSize, imageSize]
					)
					// Get the values array from the pixels tensor
					.dataSync()
				}
			);
			// Add new array to trainingData.pixels.input to store the pixel values for the image
			pixelsArray = [];
			// Loop through each value in the pixels array
			// The whole pixels array is not pushed on at once because the array format will be incompatible
			pixels.forEach(
				// Add color value to the corresponding image's trainingData.pixels.input array
				(element) => pixelsArray.push(element)
			);
			trainingData.pixels.input.push(pixelsArray);

			outputValues = new Array(numParameters).fill(1);
			trainingData.pixels.output.push(outputValues);


			// Uncaught Error: Constructing tensor of shape (92160) should match the length of values (46095)
			const generated =
			tf.tidy(
				() => generator.model.predict(parameters.display).dataSync()
			);
			const generatedArray = [];
			generated.forEach(
				(element) => generatedArray.push(element)
			);
			trainingData.pixels.input.push(generatedArray);

			outputValues = new Array(numParameters).fill(-1);
			trainingData.pixels.output.push(outputValues);
		}

		// Create a tensor from the pixel values of the training data and store it in trainingData.tensor.input
		trainingData.tensor.input = tf.tensor(trainingData.pixels.input);
		trainingData.tensor.output = tf.tensor(trainingData.pixels.output);
	}

	generateTrainingData();

	// trainingData.tensor.output = tf.ones([numTrainingImages, 6]);
	// trainingData.tensor.output.dtype = "float32";

	// Pick a random image from the testing data set to test the network on
	var index = Math.floor(Math.random() * trainingData.pixels.input.length);
	// Create image tensor from input image pixel data
	const input = tf.tensor(trainingData.pixels.input[index], [imageSize, imageSize, 3]);
	// Set input image tensor dtype to "int32"
	input.dtype = "int32";
	// Display input image on the input canvas, then dispose of the input tensor
	tf.toPixels(input, canvas).then(() => input.dispose());

	// Define training function for class-matching neural network - this will be executed iteratively
	function train() {
		if (iteration % 10 == 0) {
			generateTrainingData();
		}

		generatorLoss = generator.calculateLoss();
		discriminatorLoss = discriminator.calculateLoss();

		if (logData) {
			console.log("Iteration " + iteration);

			console.log("Generator network loss");
			generatorLoss.print();
			// Minimize the error/cost calculated by the loss calculation funcion using the optimization function

			console.log("Discriminator network loss");
			discriminatorLoss.print();

			console.log("Training data");
			console.log(trainingData);

			// Print TensorFlow.js memory information to console, including the number of tensors stored in memory (for debugging purposes)
			console.log("Memory information");
			console.log(tf.memory());
		}
		document.querySelector("#iteration").innerHTML = "Iteration &#8226; " + iteration;
		document.querySelector("#generator-loss").innerHTML = "Generator &#8226; " +
		generatorLoss.
		dataSync()[0].
		toFixed(2);
		document.querySelector("#discriminator-loss").innerHTML = "Discriminator &#8226; " +
		discriminatorLoss.
		dataSync()[0].
		toFixed(2);

		generatorLoss.dispose();
		discriminatorLoss.dispose();

		generator.optimizer.minimize(generator.calculateLoss);
		discriminator.optimizer.minimize(discriminator.calculateLoss);

		// All this is just display code
		// Calculate autoencoder output from original image
		const output =
		// Wrap output calculation in a tf.tidy() to remove intermediate tensors after the calculation is complete
		tf.tidy(
			() => {
				// Decode the low-dimensional representation of the input data created by the encoder
				return generator.model.predict(parameters.display)
				// Clip pixel values to a 0 - 255 (int32) range
				.clipByValue(0, 1)
				// Reshape the output tensor into an image format (W * L * 3)
				.reshape(
					[imageSize, imageSize, 3]
				)
			}
		);
		// output.dtype = "int32";

		const discriminatorOutput =
		tf.tidy(
			() => {
				return discriminator.model.predict(
					generator.model.predict(parameters.display)
				);
			}
		);

		if (logData) {
			console.log("Generator output");
			output.print();

			console.log("Discriminator output");
			discriminatorOutput.print();
		}

		// Display the output tensor on the output canvas, then dispose the tensor
		tf.toPixels(output, canvas).then(() => output.dispose());
		discriminatorOutput.dispose();

		iteration ++;
	}
	// Set an interval of 100 milliseconds to repeat the train() function
	var interval = window.setInterval(train, 10);
}
// Load source paths for training data images (this must be done after the image elements are created and the onload function is defined)
// Loop through each image element
for (var i = 0; i < numTrainingImages; i ++) {
	// Set the corresponding source for the image
	trainingData.images[i].src = "./training-data/" + (i + 1) + ".png";
}
