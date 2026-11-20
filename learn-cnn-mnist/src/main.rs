struct Tensor{
	data: Vec<f32>, // A buffer for storing tensor values.
	shape: Vec<usize>, // The shape of the tensor.
}

impl Tensor{

	fn new(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
		// Checks if length of data and tensor shape matches.
		// If they match, we can construct and return the Tensor instance.
			
		let num_elements: usize = shape.iter().product();

		assert_eq!(
			data.len(),
			num_elements,
			"Data length must match tensor shape!"
		);

		Tensor {data, shape}
	}

	fn flat_index(&self, indices: &[usize]) -> usize{
		// We take the given coordinate value and convert it
		// Into a flat index. We also perform some assertion
		// For ensuring the shape of the coordinate and the
		// Shape of the Tensor matches, as well as to make sure
		// The coordinate values are valid under the shape.
		
		assert_eq!(indices.len(), self.shape.len());
		let mut flat = 0;
		let mut stride = 1;

		for dim in (0..self.shape.len()).rev(){
			assert!(indices[dim] < self.shape[dim]);
			flat += indices[dim] * stride;
			stride *= self.shape[dim];
		}

		flat
	}

	fn get(&self, indices: &[usize]) -> f32{
		// Simply using the flat_index function to
		// Get the value at a coordinate from a Tensor.
		
		let index = self.flat_index(indices);
		self.data[index]
	}

	fn set(&mut self, indices: &[usize], value: f32){
		// Simply using the flat_index function to
		// Set the value at an index position of a Tensor.
		
		let index = self.flat_index(indices);
		self.data[index] = value;
	}


}

fn conv2d(input: &Tensor, kernel: &Tensor)-> Tensor{
		// The convolution forward pass.
		// Since this is a 2D conv, we assert that the input and kernel
		// Are both 2D. Then we make sure the kernel is smaller than the
		// Input. Here stride is 1 and padding is 0. Which is used for
		// Output shape calculations. Then it is the simple operation of
		// Sliding the kernel over the input, getting the sum of products,
		// And storing them in the output vector.
		
		assert_eq!(input.shape.len(),2);
		assert_eq!(kernel.shape.len(),2);

		let input_h=input.shape[0];
		let input_w=input.shape[1];

		let kernel_h=kernel.shape[0];
		let kernel_w=kernel.shape[1];

		assert!(kernel_h <= input_h && kernel_w <= input_w);

	let output_h = input_h - kernel_h + 1;
	let output_w = input_w - kernel_w + 1;

	let mut output = Tensor::new(
		vec![0.0; output_h * output_w],
		vec![output_h, output_w],
	);

	for i in 0..output_h{
		for j in 0..output_w{
			let mut sum = 0.0;

			for u in 0..kernel_h{
				for v in 0..kernel_w{
					let input_value = input.get(&[i + u, j + v]);
					let kernel_value = kernel.get(&[u, v]);
					sum += input_value * kernel_value;
				}
			}

			output.set(&[i, j], sum);
		}
	}

	output
	
	}

fn main() {
	// Testing the implementation.
	
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ],
        vec![3, 3],
    );

    let kernel = Tensor::new(
        vec![
            1.0, 2.0,
            3.0, 4.0,
        ],
        vec![2, 2],
    );

    let output = conv2d(&input, &kernel);

    println!("Output shape: {:?}", output.shape);
    println!("Output data: {:?}", output.data);
}
