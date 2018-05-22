package data

import (
	"testing"
	"fmt"
	"runtime"
)

func Test(t *testing.T) {
	_, filename, _, ok := runtime.Caller(1)
	if ok {
		fmt.Println(filename)
	}


	fmt.Println(`Done`)
}
