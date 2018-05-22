package main

import (
	"log"
	"math"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/shtmpl/learning"
	"github.com/shtmpl/learning/program/classification.digit/data"
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

func evaluate(network *core.Network, examples []core.Example) int {
	result := 0
	for _, example := range examples {
		actual, _ := maxFloat64(network.Feedforward(example.Input)...)
		expected, _ := maxFloat64(example.Output...)

		if actual != -1 && expected != -1 && actual == expected {
			result++
		}
	}

	return result
}

func main() {
	training, validation, _, err := data.Load()
	if err != nil {
		log.Fatal(err)
	}

	network := core.NewNetwork(784, 30, 10)

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}

	p.Title.Text = "Accuracy (%) on the validation data"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Accuracy"

	points := make(plotter.XYs, 30)

	for epoch := 0; epoch < 30; epoch++ {
		start := time.Now()
		network.LearnStochastically(core.CrossEntropyCost, 0.5, 10, training[:1000])

		elapsed := time.Since(start)

		accuracy := evaluate(network, validation)
		log.Printf("Epoch %d: %d / %d. Elapsed time: %v\n", epoch, accuracy, len(validation), elapsed)

		points[epoch].X = float64(epoch)
		points[epoch].Y = float64(accuracy) / float64(len(validation)) * 100
	}

	err = plotutil.AddLinePoints(p, points)
	if err != nil {
		log.Fatal(err)
	}

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		log.Fatal(err)
	}

	log.Println(`Done`)
}
