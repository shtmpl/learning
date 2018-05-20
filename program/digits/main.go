package main

import (
	"compress/gzip"
	"bytes"
	"encoding/binary"
	"os"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/shtmpl/learning"
)

const (
	MNIST_TRAINING_SET_LABELS = `program/digits/resources/train-labels-idx1-ubyte.gz`
	MNIST_TRAINING_SET_IMAGES = `program/digits/resources/train-images-idx3-ubyte.gz`

	MNIST_TEST_SET_LABELS = `program/digits/resources/t10k-labels-idx1-ubyte.gz`
	MNIST_TEST_SET_IMAGES = `program/digits/resources/t10k-images-idx3-ubyte.gz`
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

func readTrainingExamples() ([]core.Example, error) {
	labels, err := readLabels(MNIST_TRAINING_SET_LABELS)
	if err != nil {
		return nil, err
	}

	images, err := readImages(MNIST_TRAINING_SET_IMAGES)
	if err != nil {
		return nil, err
	}

	enough := math.Min(float64(len(labels)), float64(len(images)))
	result := make([]core.Example, int(enough))
	for i := range result {
		result[i] = wrapExample(labels[i], images[i])
	}

	return result, nil
}

func readTestExamples() ([]core.Example, error) {
	labels, err := readLabels(MNIST_TEST_SET_LABELS)
	if err != nil {
		return nil, err
	}

	images, err := readImages(MNIST_TEST_SET_IMAGES)
	if err != nil {
		return nil, err
	}

	enough := math.Min(float64(len(labels)), float64(len(images)))
	result := make([]core.Example, int(enough))
	for i := range result {
		result[i] = wrapExample(labels[i], images[i])
	}

	return result, nil
}

func maxFloat64(xs ...float64) (int, float64) {
	index, max := -1, math.Inf(-1)
	for i := 0; i < len(xs); i++ {
		if max < xs[i] {
			index, max = i, xs[i]
		}
	}

	return index, max
}

func evaluate(network *core.Network, examples []core.Example) (int, int) {
	result := 0
	for _, example := range examples {
		//fmt.Printf("%.9f\n", net.Feedforward(example.Input))
		//fmt.Printf("%.9f\n", example.X)
		//fmt.Println()

		actual, _ := maxFloat64(network.Feedforward(example.Input)...)
		expected, _ := maxFloat64(example.Output...)

		if actual != -1 && expected != -1 && actual == expected {
			result++
		}
	}

	return result, len(examples)
}

func evaluateStrictly(tolerance float64, network *core.Network, examples []core.Example) (int, int) {
	result := 0
	for _, example := range examples {
		actual, v := maxFloat64(network.Feedforward(example.Input)...)
		expected, x := maxFloat64(example.Output...)

		if actual != -1 && expected != -1 && actual == expected && math.Abs(v-x) < tolerance {
			result++
		}
	}

	return result, len(examples)
}

func main() {
	data, err := readTrainingExamples()
	if err != nil {
		log.Fatal(err)
	}

	//test, err := readTestExamples()
	//if err != nil {
	//	log.Fatal(err)
	//}

	training, validation := data[:50000], data[50000:]

	network := core.NewNetwork(784, 30, 10)

	for epoch := 0; epoch < 30; epoch++ {
		start := time.Now()
		network.LearnStochastically(core.CrossEntropyCost, 0.5, 10, training)

		elapsed := time.Since(start)

		success, total := evaluate(network, validation)
		//success, total := evaluateStrictly(0.01, network, validation)
		log.Printf("Epoch %d: %d / %d. Elapsed time: %v\n", epoch, success, total, elapsed)
	}

	log.Println(`Done`)
}

func show(example core.Example) {
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("%v:\n", example.Output))
	for i := 0; i < len(example.Input); i++ {
		if i%28 == 0 {
			buffer.WriteRune('\n')
		}

		if example.Input[i] == 0.0 {
			buffer.WriteRune(' ')
		} else {
			buffer.WriteRune('.')
		}
	}

	fmt.Println(buffer.String())
}
