package main

import (
	"log"

	"fmt"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
)

func run() {
	cfg := pixelgl.WindowConfig{
		Title:  "Pixel Rocks!",
		Bounds: pixel.R(0, 0, 1024, 768),
		VSync:  true,
	}

	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		log.Fatal(err)
	}

	win.Clear(colornames.Skyblue)

	r := pixel.MakePictureData(pixel.R(10, 10, 42, 42))
	fmt.Println(r.Color(pixel.V(10, 10)))
	sprite := pixel.NewSprite(r, r.Bounds())

	sprite.Draw(win, pixel.IM)

	for !win.Closed() {
		win.Update()
	}

}

func main() {
	pixelgl.Run(run)

}
