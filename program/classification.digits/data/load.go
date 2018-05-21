package data

import (
	"encoding/binary"
	"compress/gzip"
	"os"

	"github.com/shtmpl/learning"
)

const (
	MNIST_TRAINING_SET_LABELS = `program/classification.digits/data/train-labels-idx1-ubyte.gz`
	MNIST_TRAINING_SET_IMAGES = `program/classification.digits/data/train-images-idx3-ubyte.gz`

	MNIST_TEST_SET_LABELS = `program/classification.digits/data/t10k-labels-idx1-ubyte.gz`
	MNIST_TEST_SET_IMAGES = `program/classification.digits/data/t10k-images-idx3-ubyte.gz`
)

func readLabels(name string) ([]byte, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}

	defer f.Close()

	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	defer r.Close()

	var magicNumber uint32
	if err := binary.Read(r, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}

	var numberOfItems uint32
	if err := binary.Read(r, binary.BigEndian, &numberOfItems); err != nil {
		return nil, err
	}

	result := make([]byte, numberOfItems)
	if err := binary.Read(r, binary.BigEndian, result); err != nil {
		return nil, err
	}

	return result, nil
}

func readImages(name string) ([][][]byte, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}

	defer f.Close()

	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	defer r.Close()

	var magicNumber uint32
	if err := binary.Read(r, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}

	var numberOfImages uint32
	if err := binary.Read(r, binary.BigEndian, &numberOfImages); err != nil {
		return nil, err
	}

	var numberOfRows uint32
	if err := binary.Read(r, binary.BigEndian, &numberOfRows); err != nil {
		return nil, err
	}

	var numberOfColumns uint32
	if err := binary.Read(r, binary.BigEndian, &numberOfColumns); err != nil {
		return nil, err
	}

	result := make([][][]byte, numberOfImages)
	for x := range result {
		rows := make([][]byte, numberOfRows)
		for i := range rows {
			columns := make([]byte, numberOfColumns)
			if err := binary.Read(r, binary.BigEndian, columns); err != nil {
				return nil, err
			}

			rows[i] = columns
		}

		result[x] = rows
	}

	return result, nil
}

func wrapExample(label byte, image [][]byte) core.Example {
	r, c := len(image), len(image[0])
	input := make([]float64, r*c)
	for i, row := range image {
		for j, x := range row {
			input[i*c+j] = float64(x) / 255
		}
	}

	x := make([]float64, 10)
	if label < 10 {
		x[uint(label)] = 1.0
	}

	return core.Example{Input: input, Output: x}
}

func min(x, y int) int {
	if x < y {
		return x
	}

	return y
}

func LoadTrainingExamples() ([]core.Example, error) {
	labels, err := readLabels(MNIST_TRAINING_SET_LABELS)
	if err != nil {
		return nil, err
	}

	images, err := readImages(MNIST_TRAINING_SET_IMAGES)
	if err != nil {
		return nil, err
	}

	enough := min(len(labels), len(images))
	result := make([]core.Example, int(enough))
	for i := range result {
		result[i] = wrapExample(labels[i], images[i])
	}

	return result, nil
}

func LoadTestExamples() ([]core.Example, error) {
	labels, err := readLabels(MNIST_TEST_SET_LABELS)
	if err != nil {
		return nil, err
	}

	images, err := readImages(MNIST_TEST_SET_IMAGES)
	if err != nil {
		return nil, err
	}

	enough := min(len(labels), len(images))
	result := make([]core.Example, int(enough))
	for i := range result {
		result[i] = wrapExample(labels[i], images[i])
	}

	return result, nil
}
