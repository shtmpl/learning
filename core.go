package core

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
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
	X     []float64
	Input []float64
}

func (net *Network) LearnIncrementally(eta float64, example Example) {
	zs := make([]*mat.Dense, net.Depth)
	zs[0] = mat.NewDense(0, 0, nil) // fake weighted output for an input layer

	as := make([]*mat.Dense, net.Depth)
	as[0] = mat.NewDense(len(example.Input), 1, example.Input)

	for layer := 1; layer < net.Depth; layer++ {
		zs[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
		zs[layer].Mul(net.Weights[layer], as[layer-1])
		zs[layer].Add(zs[layer], net.Biases[layer])

		as[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
		as[layer].Apply(func(_, _ int, x float64) float64 { return Sigmoid(x) }, zs[layer])
	}

	dws := make([]*mat.Dense, net.Depth)
	dws[0] = mat.NewDense(0, 0, nil)

	dbs := make([]*mat.Dense, net.Depth)
	dbs[0] = mat.NewDense(0, 0, nil)

	ds := make([]*mat.Dense, net.Depth)
	ds[0] = mat.NewDense(0, 0, nil)

	L := net.Depth - 1

	sp := mat.NewDense(net.Sizes[L], 1, nil)
	sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[L])

	ds[L] = mat.NewDense(net.Sizes[L], 1, nil)
	ds[L].Sub(as[L], mat.NewDense(len(example.X), 1, example.X))
	ds[L].MulElem(ds[L], sp)

	wr, wc := net.Weights[L].Dims()
	dws[L] = mat.NewDense(wr, wc, nil)
	dws[L].Mul(ds[L], as[L-1].T())
	dws[L].Scale(eta, dws[L])

	br, bc := net.Biases[L].Dims()
	dbs[L] = mat.NewDense(br, bc, nil)
	dbs[L].Copy(ds[L])
	dbs[L].Scale(eta, dbs[L])

	for layer := L - 1; layer > 0; layer-- {
		sp := mat.NewDense(net.Sizes[layer], 1, nil)
		sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[layer])

		ds[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
		ds[layer].Mul(net.Weights[layer+1].T(), ds[layer+1])
		ds[layer].MulElem(ds[layer], sp)

		wr, wc := net.Weights[layer].Dims()
		dws[layer] = mat.NewDense(wr, wc, nil)
		dws[layer].Mul(ds[layer], as[layer-1].T())

		br, bc := net.Biases[layer].Dims()
		dbs[layer] = mat.NewDense(br, bc, nil)
		dbs[layer].Copy(ds[layer])
	}

	for layer := 1; layer < net.Depth; layer++ {
		dws[layer].Scale(eta, dws[layer])
		net.Weights[layer].Sub(net.Weights[layer], dws[layer])

		dbs[layer].Scale(eta, dbs[layer])
		net.Biases[layer].Sub(net.Biases[layer], dbs[layer])
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

func (net *Network) learnStochasticallyWithSequentialBatchProcessing(eta float64, size int, examples []Example) {
	shuffle(examples)

	for _, batch := range batch(size, examples) {
		nws := make([]*mat.Dense, net.Depth)
		nws[0] = mat.NewDense(0, 0, nil)

		nbs := make([]*mat.Dense, net.Depth)
		nbs[0] = mat.NewDense(0, 0, nil)

		for layer := 1; layer < net.Depth; layer++ {
			wr, wc := net.Weights[layer].Dims()
			nws[layer] = mat.NewDense(wr, wc, nil)

			br, bc := net.Biases[layer].Dims()
			nbs[layer] = mat.NewDense(br, bc, nil)
		}

		for _, example := range batch {
			zs := make([]*mat.Dense, net.Depth)
			zs[0] = mat.NewDense(0, 0, nil) // fake weighted output for an input layer

			as := make([]*mat.Dense, net.Depth)
			as[0] = mat.NewDense(len(example.Input), 1, example.Input)

			for layer := 1; layer < net.Depth; layer++ {
				zs[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
				zs[layer].Mul(net.Weights[layer], as[layer-1])
				zs[layer].Add(zs[layer], net.Biases[layer])

				as[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
				as[layer].Apply(func(_, _ int, x float64) float64 { return Sigmoid(x) }, zs[layer])
			}

			ds := make([]*mat.Dense, net.Depth)
			ds[0] = mat.NewDense(0, 0, nil)

			dws := make([]*mat.Dense, net.Depth)
			dws[0] = mat.NewDense(0, 0, nil)

			dbs := make([]*mat.Dense, net.Depth)
			dbs[0] = mat.NewDense(0, 0, nil)

			L := net.Depth - 1

			sp := mat.NewDense(net.Sizes[L], 1, nil)
			sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[L])

			ds[L] = mat.NewDense(net.Sizes[L], 1, nil)
			ds[L].Sub(as[L], mat.NewDense(len(example.X), 1, example.X))
			ds[L].MulElem(ds[L], sp)

			wr, wc := net.Weights[L].Dims()
			dws[L] = mat.NewDense(wr, wc, nil)
			dws[L].Mul(ds[L], as[L-1].T())

			br, bc := net.Biases[L].Dims()
			dbs[L] = mat.NewDense(br, bc, nil)
			dbs[L].Copy(ds[L])

			for layer := L - 1; layer > 0; layer-- {
				sp := mat.NewDense(net.Sizes[layer], 1, nil)
				sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[layer])

				ds[layer] = mat.NewDense(net.Sizes[layer], 1, nil)
				ds[layer].Mul(net.Weights[layer+1].T(), ds[layer+1])
				ds[layer].MulElem(ds[layer], sp)

				wr, wc := net.Weights[layer].Dims()
				dws[layer] = mat.NewDense(wr, wc, nil)
				dws[layer].Mul(ds[layer], as[layer-1].T())

				br, bc := net.Biases[layer].Dims()
				dbs[layer] = mat.NewDense(br, bc, nil)
				dbs[layer].Copy(ds[layer])
			}

			for layer := 1; layer < net.Depth; layer++ {
				nws[layer].Add(nws[layer], dws[layer])
				nbs[layer].Add(nbs[layer], dbs[layer])
			}
		}

		for layer := 1; layer < net.Depth; layer++ {
			nws[layer].Scale(eta/float64(len(batch)), nws[layer])
			net.Weights[layer].Sub(net.Weights[layer], nws[layer])

			nbs[layer].Scale(eta/float64(len(batch)), nbs[layer])
			net.Biases[layer].Sub(net.Biases[layer], nbs[layer])
		}
	}
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

func (net *Network) learnStochasticallyWithMatrixBatchProcessing(eta float64, size int, examples []Example) {
	shuffle(examples)

	for _, batch := range batch(size, examples) {
		nws := make([]*mat.Dense, net.Depth)
		nws[0] = mat.NewDense(0, 0, nil)

		nbs := make([]*mat.Dense, net.Depth)
		nbs[0] = mat.NewDense(0, 0, nil)

		for layer := 1; layer < net.Depth; layer++ {
			wr, wc := net.Weights[layer].Dims()
			nws[layer] = mat.NewDense(wr, wc, nil)

			br, bc := net.Biases[layer].Dims()
			nbs[layer] = mat.NewDense(br, bc, nil)
		}

		m, inputLen, outputLen := len(batch), len(batch[0].Input), len(batch[0].X)

		x := mat.NewDense(inputLen, m, nil)
		y := mat.NewDense(outputLen, m, nil)
		for i, example := range batch {
			x.SetCol(i, example.Input)
			y.SetCol(i, example.X)
		}

		zs := make([]*mat.Dense, net.Depth)
		zs[0] = mat.NewDense(0, 0, nil) // fake weighted output for an input layer

		as := make([]*mat.Dense, net.Depth)
		as[0] = mat.DenseCopyOf(x)

		for layer := 1; layer < net.Depth; layer++ {
			zs[layer] = mat.NewDense(net.Sizes[layer], m, nil)
			zs[layer].Mul(net.Weights[layer], as[layer-1])
			zs[layer].Add(zs[layer], extendCols(m, net.Biases[layer]))

			as[layer] = mat.NewDense(net.Sizes[layer], m, nil)
			as[layer].Apply(func(_, _ int, x float64) float64 { return Sigmoid(x) }, zs[layer])
		}

		ds := make([]*mat.Dense, net.Depth)
		ds[0] = mat.NewDense(0, 0, nil)

		dws := make([]*mat.Dense, net.Depth)
		dws[0] = mat.NewDense(0, 0, nil)

		dbs := make([]*mat.Dense, net.Depth)
		dbs[0] = mat.NewDense(0, 0, nil)

		L := net.Depth - 1

		sp := mat.NewDense(net.Sizes[L], m, nil)
		sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[L])

		ds[L] = mat.NewDense(net.Sizes[L], m, nil)
		ds[L].Sub(as[L], y)
		ds[L].MulElem(ds[L], sp)

		wr, wc := net.Weights[L].Dims()
		dws[L] = mat.NewDense(wr, wc, nil)
		for j := 0; j < m; j++ {
			dw := mat.NewDense(wr, wc, nil)
			dw.Mul(ds[L].ColView(j), as[L-1].ColView(j).T())
			dws[L].Add(dws[L], dw)
		}

		br, bc := net.Biases[L].Dims()
		dbs[L] = mat.NewDense(br, bc, nil)
		for j := 0; j < m; j++ {
			dbs[L].Add(dbs[L], ds[L].ColView(j))
		}

		for layer := L - 1; layer > 0; layer-- {
			sp := mat.NewDense(net.Sizes[layer], m, nil)
			sp.Apply(func(_, _ int, x float64) float64 { return SigmoidPrime(x) }, zs[layer])

			ds[layer] = mat.NewDense(net.Sizes[layer], m, nil)
			ds[layer].Mul(net.Weights[layer+1].T(), ds[layer+1])
			ds[layer].MulElem(ds[layer], sp)

			wr, wc := net.Weights[layer].Dims()
			dws[layer] = mat.NewDense(wr, wc, nil)
			for j := 0; j < m; j++ {
				dw := mat.NewDense(wr, wc, nil)
				dw.Mul(ds[layer].ColView(j), as[layer-1].ColView(j).T())
				dws[layer].Add(dws[layer], dw)
			}

			br, bc := net.Biases[layer].Dims()
			dbs[layer] = mat.NewDense(br, bc, nil)
			for j := 0; j < m; j++ {
				dbs[layer].Add(dbs[layer], ds[layer].ColView(j))
			}
		}

		for layer := 1; layer < net.Depth; layer++ {
			dws[layer].Scale(eta/float64(m), dws[layer])
			net.Weights[layer].Sub(net.Weights[layer], dws[layer])

			dbs[layer].Scale(eta/float64(m), dbs[layer])
			net.Biases[layer].Sub(net.Biases[layer], dbs[layer])
		}
	}
}

func (net *Network) LearnStochastically(eta float64, size int, examples []Example) {
	//net.learnStochasticallyWithSequentialBatchProcessing(eta, size, examples)
	net.learnStochasticallyWithMatrixBatchProcessing(eta, size, examples)
}

func feedforward(w, a, b *mat.Dense, f func(float64) float64) *mat.Dense {
	rows, _ := w.Dims()
	result := mat.NewDense(rows, 1, nil)

	result.Mul(w, a)
	result.Add(result, b)
	result.Apply(func(_, _ int, v float64) float64 { return f(v) }, result)

	return result
}

func (net *Network) Feedforward(input []float64) []float64 {
	x := mat.NewDense(len(input), 1, input)
	for layer := 1; layer < net.Depth; layer++ {
		w, b := net.Weights[layer], net.Biases[layer]

		x = feedforward(w, x, b, Sigmoid)
	}

	r, _ := x.Dims()
	result := make([]float64, r)

	return mat.Col(result, 0, x)
}
