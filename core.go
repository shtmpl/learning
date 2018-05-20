package core

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

type Network struct {
	Depth   int
	Sizes   []int
	Weights []*mat.Dense
	Biases  []*mat.Dense
}

func randN(n int) []float64 {
	result := make([]float64, n)

	for i := range result {
		result[i] = rand.NormFloat64()
	}

	return result
}

func randMatrix(r, c int) *mat.Dense {
	return mat.NewDense(r, c, randN(r*c))
}

func NewNetwork(sizes ...int) *Network {
	ws := make([]*mat.Dense, len(sizes))
	ws[0] = mat.NewDense(0, 0, nil)
	for i := 1; i < len(sizes); i++ {
		ws[i] = randMatrix(sizes[i], sizes[i-1])
	}

	bs := make([]*mat.Dense, len(sizes))
	bs[0] = mat.NewDense(0, 0, nil)
	for i := 1; i < len(sizes); i++ {
		bs[i] = randMatrix(sizes[i], 1)
	}

	return &Network{
		Depth:   len(sizes),
		Sizes:   sizes,
		Weights: ws,
		Biases:  bs,
	}
}

type Example struct {
	Input  []float64
	Output []float64
}

type Gradient struct {
	Weights []*mat.Dense
	Biases  []*mat.Dense
}

type Cost struct {
	Derivative func(z, a, y *mat.Dense) *mat.Dense
}

var QuadraticCost = Cost{
	Derivative: func(z, a, y *mat.Dense) *mat.Dense {
		r, c := y.Dims()

		dz := mat.NewDense(r, c, nil)
		dz.Apply(func(_, _ int, x float64) float64 { return SigmoidDerivative(x) }, z)

		result := mat.NewDense(r, c, nil)
		result.Sub(a, y)
		result.MulElem(result, dz)

		return result
	},
}

var CrossEntropyCost = Cost{
	Derivative: func(_, a, y *mat.Dense) *mat.Dense {
		r, c := y.Dims()

		result := mat.NewDense(r, c, nil)
		result.Sub(a, y)

		return result
	},
}

func (network *Network) LearnIncrementally(eta float64, example Example) {
	zs := make([]*mat.Dense, network.Depth)
	zs[0] = mat.NewDense(0, 0, nil) // fake weighted output for an input layer

	as := make([]*mat.Dense, network.Depth)
	as[0] = mat.NewDense(len(example.Input), 1, example.Input)

	for layer := 1; layer < network.Depth; layer++ {
		zs[layer] = mat.NewDense(network.Sizes[layer], 1, nil)
		zs[layer].Mul(network.Weights[layer], as[layer-1])
		zs[layer].Add(zs[layer], network.Biases[layer])

		as[layer] = mat.NewDense(network.Sizes[layer], 1, nil)
		as[layer].Apply(func(_, _ int, x float64) float64 { return Sigmoid(x) }, zs[layer])
	}

	ds := make([]*mat.Dense, network.Depth)
	ds[0] = mat.NewDense(0, 0, nil)

	dws := make([]*mat.Dense, network.Depth)
	dws[0] = mat.NewDense(0, 0, nil)

	dbs := make([]*mat.Dense, network.Depth)
	dbs[0] = mat.NewDense(0, 0, nil)

	L := network.Depth - 1

	sp := mat.NewDense(network.Sizes[L], 1, nil)
	sp.Apply(func(_, _ int, x float64) float64 { return SigmoidDerivative(x) }, zs[L])

	ds[L] = mat.NewDense(network.Sizes[L], 1, nil)
	ds[L].Sub(as[L], mat.NewDense(len(example.Output), 1, example.Output))
	ds[L].MulElem(ds[L], sp)

	wr, wc := network.Weights[L].Dims()
	dws[L] = mat.NewDense(wr, wc, nil)
	dws[L].Mul(ds[L], as[L-1].T())

	br, bc := network.Biases[L].Dims()
	dbs[L] = mat.NewDense(br, bc, nil)
	dbs[L].Copy(ds[L])

	for layer := L - 1; layer > 0; layer-- {
		sp := mat.NewDense(network.Sizes[layer], 1, nil)
		sp.Apply(func(_, _ int, x float64) float64 { return SigmoidDerivative(x) }, zs[layer])

		ds[layer] = mat.NewDense(network.Sizes[layer], 1, nil)
		ds[layer].Mul(network.Weights[layer+1].T(), ds[layer+1])
		ds[layer].MulElem(ds[layer], sp)

		wr, wc := network.Weights[layer].Dims()
		dws[layer] = mat.NewDense(wr, wc, nil)
		dws[layer].Mul(ds[layer], as[layer-1].T())

		br, bc := network.Biases[layer].Dims()
		dbs[layer] = mat.NewDense(br, bc, nil)
		dbs[layer].Copy(ds[layer])
	}

	for layer := 1; layer < network.Depth; layer++ {
		dws[layer].Scale(eta, dws[layer])
		network.Weights[layer].Sub(network.Weights[layer], dws[layer])

		dbs[layer].Scale(eta, dbs[layer])
		network.Biases[layer].Sub(network.Biases[layer], dbs[layer])
	}
}

func shuffle(examples []Example) {
	for i := range examples {
		j := rand.Intn(i + 1)
		examples[i], examples[j] = examples[j], examples[i]
	}
}

func min(x, y int) int {
	if x < y {
		return x
	}

	return y
}

func batch(size int, examples []Example) [][]Example {
	result := make([][]Example, 0, size)
	for i := 0; i < len(examples); i += size {
		result = append(result, examples[i:min(i+size, len(examples))])
	}

	return result
}

func ones(enough int) []float64 {
	result := make([]float64, enough)
	for i := range result {
		result[i] = 1.0
	}

	return result
}

func extendCols(times int, m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, times*c, nil)
	for i := 0; i < times; i++ {
		result.Mul(m, mat.NewDense(c, times*c, ones(times*c*c)))
	}

	return result
}

func (network *Network) processBatch(cost Cost, eta float64, examples []Example) *Gradient {
	m, lenInput, lenOutput := len(examples), len(examples[0].Input), len(examples[0].Output)

	x := mat.NewDense(lenInput, m, nil)
	y := mat.NewDense(lenOutput, m, nil)
	for i, example := range examples {
		x.SetCol(i, example.Input)
		y.SetCol(i, example.Output)
	}

	zs := make([]*mat.Dense, network.Depth)
	zs[0] = mat.NewDense(0, 0, nil) // fake weighted output for an input layer

	as := make([]*mat.Dense, network.Depth)
	as[0] = mat.DenseCopyOf(x)

	for layer := 1; layer < network.Depth; layer++ {
		zs[layer] = mat.NewDense(network.Sizes[layer], m, nil)
		zs[layer].Mul(network.Weights[layer], as[layer-1])
		zs[layer].Add(zs[layer], extendCols(m, network.Biases[layer]))

		as[layer] = mat.NewDense(network.Sizes[layer], m, nil)
		as[layer].Apply(func(_, _ int, x float64) float64 { return Sigmoid(x) }, zs[layer])
	}

	ds := make([]*mat.Dense, network.Depth)
	ds[0] = mat.NewDense(0, 0, nil)

	result := &Gradient{
		Weights: make([]*mat.Dense, network.Depth),
		Biases:  make([]*mat.Dense, network.Depth),
	}

	result.Weights[0] = mat.NewDense(0, 0, nil)

	result.Biases[0] = mat.NewDense(0, 0, nil)

	L := network.Depth - 1

	ds[L] = cost.Derivative(zs[L], as[L], y)

	wr, wc := network.Weights[L].Dims()
	result.Weights[L] = mat.NewDense(wr, wc, nil)
	for j := 0; j < m; j++ {
		dw := mat.NewDense(wr, wc, nil)
		dw.Mul(ds[L].ColView(j), as[L-1].ColView(j).T())
		result.Weights[L].Add(result.Weights[L], dw)
	}

	br, bc := network.Biases[L].Dims()
	result.Biases[L] = mat.NewDense(br, bc, nil)
	for j := 0; j < m; j++ {
		result.Biases[L].Add(result.Biases[L], ds[L].ColView(j))
	}

	for layer := L - 1; layer > 0; layer-- {
		dz := mat.NewDense(network.Sizes[layer], m, nil)
		dz.Apply(func(_, _ int, x float64) float64 { return SigmoidDerivative(x) }, zs[layer])

		ds[layer] = mat.NewDense(network.Sizes[layer], m, nil)
		ds[layer].Mul(network.Weights[layer+1].T(), ds[layer+1])
		ds[layer].MulElem(ds[layer], dz)

		wr, wc := network.Weights[layer].Dims()
		result.Weights[layer] = mat.NewDense(wr, wc, nil)
		for j := 0; j < m; j++ {
			dw := mat.NewDense(wr, wc, nil)
			dw.Mul(ds[layer].ColView(j), as[layer-1].ColView(j).T())
			result.Weights[layer].Add(result.Weights[layer], dw)
		}

		br, bc := network.Biases[layer].Dims()
		result.Biases[layer] = mat.NewDense(br, bc, nil)
		for j := 0; j < m; j++ {
			result.Biases[layer].Add(result.Biases[layer], ds[layer].ColView(j))
		}
	}

	return result
}

func (network *Network) LearnStochastically(cost Cost, eta float64, size int, examples []Example) {
	shuffle(examples)

	for _, batch := range batch(size, examples) {
		g := network.processBatch(cost, eta, batch)

		for layer := 0; layer < network.Depth; layer++ {
			g.Weights[layer].Scale(eta/float64(len(batch)), g.Weights[layer])
			network.Weights[layer].Sub(network.Weights[layer], g.Weights[layer])

			g.Biases[layer].Scale(eta/float64(len(batch)), g.Biases[layer])
			network.Biases[layer].Sub(network.Biases[layer], g.Biases[layer])
		}
	}
}

func feedforward(w, a, b *mat.Dense, f func(float64) float64) *mat.Dense {
	rows, _ := w.Dims()
	result := mat.NewDense(rows, 1, nil)

	result.Mul(w, a)
	result.Add(result, b)
	result.Apply(func(_, _ int, v float64) float64 { return f(v) }, result)

	return result
}

func (network *Network) Feedforward(input []float64) []float64 {
	x := mat.NewDense(len(input), 1, input)
	for layer := 1; layer < network.Depth; layer++ {
		w, b := network.Weights[layer], network.Biases[layer]

		x = feedforward(w, x, b, Sigmoid)
	}

	r, _ := x.Dims()
	result := make([]float64, r)

	return mat.Col(result, 0, x)
}
