package main

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/shtmpl/learning"
	"github.com/shtmpl/learning/program/classification.digits/data"
)

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
	examples, err := data.LoadTrainingExamples()
	if err != nil {
		log.Fatal(err)
	}

	//test, err := data.LoadTestExamples()
	//if err != nil {
	//	log.Fatal(err)
	//}

	training, validation := examples[:50000], examples[50000:]

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
