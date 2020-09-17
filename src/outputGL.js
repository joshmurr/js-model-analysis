export default class OutputGL {
  constructor(canvas) {
    this._gl = canvas.getContext("webgl2")

    if (!this._gl) {
      console.error("No WebGL2 support in your chosen browser!")
    }

    this._vsSource = `#version 300 es
    in vec4 a_position;

    void main() {
      gl_Position = a_position;
    }
    `

    this._fsSource = `#version 300 es
    precision highp float;
     
    out vec4 outColor;
     
    void main() {
      outColor = vec4(1, 0, 0.5, 1);
    }
    `
    this._verts = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
    this._textureCoordinates = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    this._normals = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    // For use with TRIANGLES
    this._indices = [0, 1, 2, 0, 2, 3]

    this._program = this.createProgram(this._gl, this._vsSource, this._fsSource)

    this._posAttrLoc = this._gl.getAttribLocation(this._program, "a_position")
    const positionBuffer = this._gl.createBuffer()
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, positionBuffer)
    this._gl.bufferData(
      this._gl.ARRAY_BUFFER,
      new Float32Array(this._verts),
      this._gl.STATIC_DRAW
    )

    this._vao = this._gl.createVertexArray()
    this._gl.bindVertexArray(this._vao)

    this._gl.enableVertexAttribArray(this._posAttrLoc)
    this._gl.vertexAttribPointer(
      this._posAttrLoc,
      2, // size
      this._gl.FLOAT,
      false, //normalise,
      0, //stride
      0 //offset
    )

    this._gl.bindVertexArray(null)
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null)
  }

  draw() {
    this._gl.viewport(0, 0, this._gl.canvas.width, this._gl.canvas.height)
    this._gl.clearColor(0, 0, 0, 0)
    this._gl.clear(this._gl.COLOR_BUFFER_BIT)

    this._gl.useProgram(this._program)
    this._gl.bindVertexArray(this._vao)
    this._gl.drawArrays(this._gl.TRIANGLES, 0, this._verts.length / 2)
  }

  createShader(gl, source, type) {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader)
      throw "Could not compile WebGL program. \n\n" + info
    }
    return shader
  }

  createProgram(gl, vsSource, fsSource) {
    const shaderProgram = gl.createProgram()
    const vertexShader = this.createShader(this._gl, vsSource, gl.VERTEX_SHADER)
    const fragmentShader = this.createShader(
      this._gl,
      fsSource,
      gl.FRAGMENT_SHADER
    )
    gl.attachShader(shaderProgram, vertexShader)
    gl.attachShader(shaderProgram, fragmentShader)
    gl.linkProgram(shaderProgram)

    if (gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
      return shaderProgram
    }

    console.error("Error creating shader program!")
    gl.deleteProgram(shaderProgram)
  }
}
