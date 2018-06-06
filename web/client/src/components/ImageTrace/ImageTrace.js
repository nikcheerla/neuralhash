import React from "react";
import "./ImageTrace.css";

import Image from "image-js";
import { toPath, toPoints } from "svg-points";
import * as simplify from "simplify-js";
import * as ImageTracer from "imagetracerjs";
import HtmlToReact, { Parser } from "html-to-react";
import { TweenLite, morphSVG, TimelineLite, SlowMo, CustomEase } from "gsap";

export default class ImageTrace extends React.Component {
  state = {
    path: <path id="data" />
  };

  componentDidMount() {
    Image.load("protected.jpeg").then(async img => {
      let data = ImageTracer.imagedataToSVG(img, {
        ltres: 30,
        qtres: 30,
        numberofcolors: 2,
        pal: [{ r: 255, b: 255, g: 255, a: 0 }, { r: 0, b: 0, g: 0, a: 1 }],
        colorsampling: 0,
        linefilter: true
      });

      let svgData = this.convertSvgGroupToPath(data);

      this.setState({
        path: <path strokeDasharray="0 0" d={svgData} id="data" />
      });

      let tl = new TimelineLite();

      let orig = document.querySelector("#data");
      let obj = {
        length: 0,
        pathLength: orig.getTotalLength()
      };

      tl.to(obj, 500, {
        length: obj.pathLength,
        onUpdate: drawLine,
        ease: SlowMo.ease.config(0.1, 0.7, false)
      });

      function drawLine() {
        orig.style.strokeDasharray = [obj.length, obj.pathLength].join(" ");
      }

      tl.to(
        "#data",
        1,
        {
          morphSVG: { shape: "#lock", shapeIndex: 20 }
        },
        "-=498"
      );
      tl.play();
    });
  }

  convertSvgGroupToPath(data) {
    let finalD = "";
    var processNodeDefinitions = new HtmlToReact.ProcessNodeDefinitions(React);
    let component = new Parser().parseWithInstructions(
      data,
      node => {
        return true;
      },
      [
        {
          shouldProcessNode: function(node) {
            return node.name === "svg";
          },
          processNode: function(node, children) {
            return children;
          }
        },
        {
          shouldProcessNode: function(node) {
            return node.name === "path";
          },
          processNode: function(node, children, index) {
            finalD += " " + node.attribs.d;
            return React.createElement("path", {
              key: index,
              d: node.attribs.d
            });
          }
        },
        {
          // Anything else
          shouldProcessNode: function(node) {
            return true;
          },
          processNode: processNodeDefinitions.processDefaultNode
        }
      ]
    );
    return finalD;
  }
  /*
      let svgData = MSQR(edge.getCanvas(), {
        width: edge.width,
        tolerance: 50,
        align: false,
        alpha: 1,
        bleed: 5, // width of bleed mask (used with multiple shapes only)
        maxShapes: 5,
        height: edge.height,
        path2D: false
      });

      let a = document.createElement("a");
      document.body.appendChild(a);
      a.style = "display: none";
      let url = window.URL.createObjectURL(await edge.toBlob());
      a.href = url;
      a.download = "path.png";
      a.click();
      window.URL.revokeObjectURL(url);

      console.log(svgData);
      this.setState({ path: toPath(svgData) });
    });
  }

  trace = image => {
    var point, nextpoint;
    let data = [];

    for (var i = 0; i <= image.data.length; i++) {
      if (image.data[i] === 255) {
        // start pathfinding
        point = { x: i % image.width, y: (i / image.width) | 0 };

        image.data[i] = 0;

        // start a line
        var line = [];
        line.push(point);
        while ((nextpoint = this.lineGobble(image, point))) {
          line.push(nextpoint);
          point = nextpoint;
        }
        data.push(line);
      }
    }
    return data;
  };

  lineGobble = (image, point) => {
    var neighbor = [
      [0, -1], // n
      [1, 0], // s
      [0, 1], // e
      [-1, 0], // w
      [-1, -1], // nw
      [1, -1], // ne
      [1, 1], // se
      [-1, 1] // sw
    ];
    var checkpoint = {};

    for (var i = 0; i < neighbor.length; i++) {
      checkpoint.x = point.x + neighbor[i][0];
      checkpoint.y = point.y + neighbor[i][1];

      var result = this.checkpixel(image, checkpoint);
      if (result) {
        return checkpoint;
      }
    }
    return false;
  };

  checkpixel = (image, point) => {
    if (0 <= point.x < image.width) {
      if (0 <= point.y < image.height) {
        // point is "in bounds"
        var index = point.y * image.width + point.x;
        if (image.data[index] === 255) {
          image.data[index] = 0;
          return true;
        }
      }
    }
    return false;
  };*/

  render() {
    return (
      <div>
        <svg height="1000" width="1000">
          {this.state.path}
          <path
            id="lock"
            style={{ visibility: "hidden" }}
            d="M400,82.531l38.168,77.336l85.346,12.401l-61.757,60.198l14.579,85.001L400,277.336l-76.336,40.132
            l14.579-85.001l-61.757-60.198l85.346-12.401L400,82.531z"
          />
        </svg>
      </div>
    );
  }
}
