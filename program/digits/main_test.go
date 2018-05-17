package main

import (
	"testing"
	"fmt"
	"math"
)

func Test(t *testing.T) {
	fmt.Println(maxFloat64(0.0, 0.1, 0.2, 0.3))
	fmt.Println(math.Inf(-1) == math.Inf(-1))

	fmt.Println(`Done`)
}
