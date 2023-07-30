# dynamic-programming-for-bio-informatics

This repository contains an implementation of the Smith-Waterman algorithm for local sequence alignment. The Smith-Waterman algorithm is used in bioinformatics to identify similarities and alignments between biological sequences such as DNA or protein sequences. It is used to infer structural, functional and evolutionary relationship between the sequences.

## Usage
- Choose to manually enter the two sequences to be aligned. (here, random initiation for sample)
- Enter the match and mismatch score for the alignment.
- The program runs on the data to generate a alignment matrix.

  ![Screenshot 2023-07-30 205805](https://github.com/iabhiroop/dynamic-programming-for-bio-informatics/assets/100859103/bc9635de-0a4c-491e-92f1-8132b6bbab90)
  
- Traceback is done to find the alignment.
- The result along with the trace back map is returned
  
  ![Screenshot 2023-07-30 205832](https://github.com/iabhiroop/dynamic-programming-for-bio-informatics/assets/100859103/8c7cf681-0adf-4452-86f9-928d5bf2b00b)

## How to run

To use this program, follow the instructions below:

1. Clone the repository to your local machine or download the source code files.

2. Make sure you have Python 3 installed on your machine.

3. Install the required dependencies by running the following command:

    ```
    pip install numpy
    ```

4. Run the program using the following command:

    ```
    python sequence_alignment.py
    ```

5. You will be prompted to choose between manually entering two sequences or generating random sequences. If you choose manual input, enter the sequences when prompted. Otherwise, the program will generate random sequences for you.

6. The program will display the input sequences, perform the Smith-Waterman algorithm for local sequence alignment, and output the aligned sequences along with a visual representation of the optimal alignment path.

## License

Feel free to use, modify, and distribute this code. Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Acknowledgments

This implementation is based on the Smith-Waterman algorithm and is inspired by bioinformatics applications. The algorithm was developed by Temple F. Smith and Michael S. Waterman.
