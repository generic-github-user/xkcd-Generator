// Main JavaScript for TensorFlow.js Deep Dream

// Define settings
// Size of input and output images in pixels (width and height)
const imageSize = 50;
// Number of images to use when training the neural network
const numTrainingImages = 20;
// Number of images to use when testing the neural network
const numTestingImages = 5;
const classTargets = tf.oneHot(tf.tensor1d(tf.ones([20]).dataSync(), "int32"), 2, 1, -1);

// Automatically generated settings and parameters
// Volume of image data, calculated by squaring imageSize to find the area of the image (total number of pixels) and multiplying by three for each color channel (RGB)
const imageVolume = (imageSize ** 2) * 3;
classTargets.dtype = "float32";

// Get information for main canvas elements
const canvas = {
	// Get information for input canvas to display randomly selected input image for the autoencoder
	"input": document.getElementById("inputCanvas"),
	// Get information for output canvas to display autoencoded representation of the original input image
	"output": document.getElementById("outputCanvas")
}
// Get context for main canvas elements
const context = {
	// Get context for input canvas
	"input": canvas.input.getContext("2d"),
	// Get context for output canvas
	"output": canvas.output.getContext("2d")
}

// Set canvas dimensions to match specified image dimensions
// Input canvas
canvas.input.width = imageSize;
canvas.input.height = imageSize;
// Output canvas
canvas.output.width = imageSize;
canvas.output.height = imageSize;

// Define encoder network with the high-level TensorFlow.js layers system
// This network takes a high-dimensional input image and reduces it to a low-dimensional "latent-space" representation
// Define encoder network layers
const classifier = {
	// Input layer with the same number of units as the volume of the input image
	"input": tf.input({shape: imageVolume}),
	// Hidden layers
	"hidden": [
		// First hidden layer - dense layer with 500 units and a relu activation function
		tf.layers.dense({units: 500, activation: "relu"}),
		// Second hidden layer - dense layer with 300 units and a relu activation function
		tf.layers.dense({units: 300, activation: "relu"}),
		// Third hidden layer - dense layer with 100 units and a relu activation function
		tf.layers.dense({units: 100, activation: "relu"})
	],
	// Wrap loss calculation function in a tf.tidy so that intermediate tensors are disposed of when the calculation is finished
	"calculateLoss": () => tf.tidy(
		// Calculate loss
		() => {
			// Evaluate the loss function given the output of the autoencoder network and the actual image
			return loss(
				classifier.model.predict(trainingData.tensor.input),
				trainingData.tensor.output
			);
		}
	)
};
// Define data flow through encoder model layers
// Output layer is a dense layer with 5 units that is calculated by applying the third ([2]) hidden layer
classifier.output = tf.layers.dense({units: 2}).apply(
	// Third hidden layer is calculated by applying the second ([1]) hidden layer
	classifier.hidden[2].apply(
		// Third hidden layer is calculated by applying the first ([0]) hidden layer
		classifier.hidden[1].apply(
			// First hidden layer is calculated by applying the input
			classifier.hidden[0].apply(
				// Encoder network input
				classifier.input
			)
		)
	)
);

// Define decoder network
// This network takes a low-dimensional "latent-space" representation of the input image (created by the encoder network) and creates a high-dimensional output image (meant to match the original input image)
// Define decoder network layers
const dreamer = {
	// Input layer with the same number of units as the output of the encoder network (the number of latent variables)
	"input": tf.input({shape: imageVolume}),
	// Hidden layers
	"hidden": [
		// First hidden layer - dense layer with 100 units and a relu activation function
		tf.layers.dense({units: Math.round(imageVolume / 2), activation: "relu"}),
		// Second hidden layer - dense layer with 300 units and a relu activation function
		tf.layers.dense({units: Math.round(imageVolume / 2), activation: "relu"}),
		// Third hidden layer - dense layer with 500 units and a relu activation function
		tf.layers.dense({units: Math.round(imageVolume / 2), activation: "relu"})
	],
	// Wrap loss calculation function in a tf.tidy so that intermediate tensors are disposed of when the calculation is finished
	"calculateLoss": () => tf.tidy(
		// Calculate loss
		() => {
			// Evaluate the loss function given the output of the autoencoder network and the actual image
			return loss(
				classifier.model.predict(trainingData.tensor.input),
				classTargets
			);
		}
	)
};
// Define data flow through decoder model layers
// Output layer is a dense layer with the same number of units as the input image/data that is calculated by applying the third ([2]) hidden layer
dreamer.output = tf.layers.dense({units: imageVolume}).apply(
	// Third hidden layer is calculated by applying the second ([1]) hidden layer
	dreamer.hidden[2].apply(
		// Third hidden layer is calculated by applying the first ([0]) hidden layer
		dreamer.hidden[1].apply(
			// First hidden layer is calculated by applying the input
			dreamer.hidden[0].apply(
				// Decoder network input
				dreamer.input
			)
		)
	)
);

// Create a new TensorFlow.js model to act as the encoder network in the autoencoder
classifier.model = tf.model(
	{
		// Set inputs to predefined encoder network input layer
		"inputs": classifier.input,
		// Set outputs to predefined encoder network outputs layer
		"outputs": classifier.output
	}
);
// Create a new model to act as the decoder network in the autoencoder
dreamer.model = tf.model(
	{
		// Set inputs to predefined decoder network input layer
		"inputs": dreamer.input,
		// Set outputs to predefined decoder network outputs layer
		"outputs": dreamer.output
	}
);

// Neural network training/optimization
// Define loss function for neural network training: Mean squared error
loss = (input, output) => input.sub(output).square().mean();
// Learning rate for optimization algorithm
const learningRate = 1;
// Optimization function for training neural networks
optimizer = tf.train.adam(learningRate);

// Create object to store training data in image, pixel, and tensor format
const trainingData = {
	// Store training data as HTML image elements
	"images": [],
	// Store training data as raw arrays of pixel data
	"pixels": [],
	// Store training data as a TensorFlow.js tensor
	"tensor": {}
}
const testingData = {
	// Store testing data as HTML image elements
	"images": [],
	// Store testing data as raw arrays of pixel data
	"pixels": [],
	// Store testing data as a TensorFlow.js tensor
	"tensor": {}
}

const paths = [
	trainingData,
	testingData
];
const numImages = [
	numTrainingImages,
	numTestingImages
];

// Add training data to trainingData.images array as an HTML image element
// Loop through each training image

for (var i = 0; i < numImages[i]; i ++) {
	for (var j = 0; j < numImages[i]; j ++) {
		// Create a new HTML image element with the specified dimensions and set current array index to this element (array.push does not work here)
		paths[i].images[j] = new Image(imageSize, imageSize);
	}
}

// Wait for last image (testing data) to load before continuing
testingData.images[testingData.images.length - 1].onload = function () {
	// Create training data from pixels of image elements
	// Create a new variable to store the data
	var pixels;
	// Loop through each training image

	for (var i = 0; i < paths.length; i ++) {
		for (var j = 0; j < numImages[i]; j ++) {
			// Create a tensor with 3 (RGB) color channels from the image element
			pixels = tf.fromPixels(paths[i].images[j], 3);
			// Resize image to the specified dimensions with resizeBilinear()
			pixels = tf.image.resizeBilinear(pixels, [imageSize, imageSize]);
			// Get the values array from the pixels tensor
			pixels = pixels.dataSync();
			// Add new array to trainingData.pixels to store the pixel values for the image
			paths[i].pixels.push([]);
			// Loop through each value in the pixels array
			// The whole pixels array is not pushed on at once because the array format will be incompatible
			pixels.forEach(
				// Add color value to the corresponding image's trainingData.pixels array
				(element) => paths[i].pixels[j].push(element)
			);
		}
	}
	// Create a tensor from the pixel values of the training data and store it in trainingData.tensor.input
	trainingData.tensor.input = tf.tensor(trainingData.pixels);
	// Create a tensor from the pixel values of the testing data and store it in testingData.tensor.input
	testingData.tensor.input = tf.tensor(testingData.pixels);

	const imageLabels = [];
	// var labels = tf.ones([10]).dataSync();
	for (var i = 0; i < 10; i ++) {
		imageLabels.push(0);
	}
	for (var i = 0; i < 10; i ++) {
		imageLabels.push(1);
	}
	trainingData.tensor.output = tf.oneHot(tf.tensor1d(imageLabels, "int32"), 2, 1, -1);
	trainingData.tensor.output.dtype = "float32";

	// Pick a random image from the testing data set to test the network on
	var index = Math.floor(Math.random() * testingData.pixels.length);
	// Create image tensor from input image pixel data
	const input = tf.tensor(testingData.pixels[index], [imageSize, imageSize, 3]);
	// Set input image tensor dtype to "int32"
	input.dtype = "int32";
	// Display input image on the input canvas, then dispose of the input tensor
	tf.toPixels(input, canvas.input).then(() => input.dispose());

	function printLoss(model) {
		// Print TensorFlow.js memory information to console, including the number of tensors stored in memory (for debugging purposes)
		console.log(tf.memory());
		// Use tidy here
		// Print current neural network loss to console
		// Calculate loss value and store it in a constant
		const currentLoss = model.calculateLoss();
		// Print loss to console
		currentLoss.print();
		// Dispose of loss value
		currentLoss.dispose();
	}

	console.log("Begin classifier network training");
	for (var i = 0; i < 100; i ++) {
		printLoss(classifier);
		// Minimize the error/cost calculated by the loss calculation funcion using the optimization function
		optimizer.minimize(classifier.calculateLoss);
	}
	console.log("End classifier network training");

	// Define training function for class-matching neural network - this will be executed iteratively
	function train() {
		// adversarial

		printLoss(classifier);
		// Minimize the error/cost calculated by the loss calculation funcion using the optimization function
		optimizer.minimize(classifier.calculateLoss);

		printLoss(dreamer);
		// Minimize the error/cost calculated by the loss calculation funcion using the optimization function
		optimizer.minimize(dreamer.calculateLoss);

		// All this is just display code
		// Calculate autoencoder output from original image
		const output =
		// Wrap output calculation in a tf.tidy() to remove intermediate tensors after the calculation is complete
		tf.tidy(
			() => {
				// Decode the low-dimensional representation of the input data created by the encoder
				return dreamer.model.predict(
					// Create a tensor from the array of pixel values for the randomly selected input image
					tf.tensor(
						[trainingData.pixels[index]]
					)
				)
				// Clip pixel values to a 0 - 255 (int32) range
				.clipByValue(0, 255)
				// Reshape the output tensor into an image format (W * L * 3)
				.reshape(
					[imageSize, imageSize, 3]
				)
			}
		);
		output.dtype = "int32";

		// Display the output tensor on the output canvas, then dispose the tensor
		tf.toPixels(output, canvas.output).then(() => output.dispose());
	}
	// Set an interval of 100 milliseconds to repeat the train() function
	var interval = window.setInterval(train, 100);
}
// Load source paths for training data images (this must be done after the image elements are created and the onload function is defined)
// Loop through each image element
for (var i = 0; i < numTrainingImages; i ++) {
	// Set the corresponding source for the image
	trainingData.images[i].src = "./training-data/" + (i + 1) + ".jpg";
}
for (var i = 0; i < numTestingImages; i ++) {
	// Set the corresponding source for the image
	testingData.images[i].src = "./testing-data/" + (i + 1) + ".jpg";
}
