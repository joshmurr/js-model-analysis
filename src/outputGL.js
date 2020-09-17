export default class OutputGL {
  constructor(canvas) {
    this._gl = canvas.getContext('webgl2');

    if (!this._gl) {
      console.error('No WebGL2 support in your chosen browser!');
    }

    this._vsSource = `#version 300 es
    in vec4 a_position;
    in vec2 a_texcoord;

    out vec2 v_texcoord;

    void main() {
      gl_Position = a_position;
      v_texcoord = a_texcoord;
    }
    `;

    this._fsSource = `#version 300 es
    precision highp float;
     
    in vec2 v_texcoord;
    uniform sampler2D u_texture;
    out vec4 outColor;

    void main() {
      outColor = texture(u_texture, (v_texcoord * vec2(0.5, -0.5)) + vec2(0.5));
    }
    `;
    this._verts = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
    this._textureCoordinates = [0, 0, 1, 0, 1, 1, 0, 1];

    this._program = this.createProgram(
      this._gl,
      this._vsSource,
      this._fsSource
    );
    this._vao = this._gl.createVertexArray();
    this._gl.bindVertexArray(this._vao);

    // Position
    this._posAttrLoc = this._gl.getAttribLocation(this._program, 'a_position');
    const positionBuffer = this._gl.createBuffer();
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, positionBuffer);
    this._gl.bufferData(
      this._gl.ARRAY_BUFFER,
      new Float32Array(this._verts),
      this._gl.STATIC_DRAW
    );
    this._gl.enableVertexAttribArray(this._posAttrLoc);
    this._gl.vertexAttribPointer(
      this._posAttrLoc,
      2, // size
      this._gl.FLOAT,
      false, //normalise,
      0, //stride
      0 //offset
    );

    // Texture Coords
    this._texAttrLoc = this._gl.getAttribLocation(this._program, 'a_texcoord');
    const texBuffer = this._gl.createBuffer();
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, texBuffer);
    this._gl.bufferData(
      this._gl.ARRAY_BUFFER,
      new Float32Array(this._verts),
      this._gl.STATIC_DRAW
    );
    this._gl.enableVertexAttribArray(this._texAttrLoc);
    this._gl.vertexAttribPointer(
      this._texAttrLoc,
      2, // size
      this._gl.FLOAT,
      false, //normalise,
      0, //stride
      0 //offset
    );

    // Texture
    //var ext = this._gl.getExtension('OES_texture_float');
    this._textureLocation = this._gl.getUniformLocation(
      this._program,
      'u_texture'
    );
    this._texture = this._gl.createTexture();
    this._gl.activeTexture(this._gl.TEXTURE0 + 0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._texture);
    this._gl.texImage2D(
      this._gl.TEXTURE_2D,
      0,
      this._gl.RGB32F,
      256,
      256,
      0,
      this._gl.RGB,
      this._gl.FLOAT,
      null
    );

    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_MAG_FILTER,
      this._gl.NEAREST
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_MIN_FILTER,
      this._gl.NEAREST
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_WRAP_S,
      this._gl.REPEAT
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_WRAP_T,
      this._gl.REPEAT
    );

    this._gl.bindVertexArray(null);
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
  }

  createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      throw 'Could not compile WebGL program. \n\n' + info;
    }
    return shader;
  }

  createProgram(gl, vsSource, fsSource) {
    const shaderProgram = gl.createProgram();
    const vertexShader = this.createShader(
      this._gl,
      vsSource,
      gl.VERTEX_SHADER
    );
    const fragmentShader = this.createShader(
      this._gl,
      fsSource,
      gl.FRAGMENT_SHADER
    );
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
      return shaderProgram;
    }

    console.error('Error creating shader program!');
    gl.deleteProgram(shaderProgram);
  }

  draw(data) {
    this._gl.viewport(0, 0, this._gl.canvas.width, this._gl.canvas.height);
    this._gl.clearColor(0, 0, 0, 0);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT);
    this._gl.useProgram(this._program);
    this._gl.bindVertexArray(this._vao);

    /*
     *this._gl.texImage2D(
     *  this._gl.TEXTURE_2D,
     *  0,
     *  this._gl.RGB32F,
     *  256,
     *  256,
     *  0,
     *  this._gl.RGB,
     *  this._gl.FLOAT,
     *  data
     *);
     */

    this._gl.texSubImage2D(
      this._gl.TEXTURE_2D,
      0,
      0,
      0,
      256,
      256,
      this._gl.RGB,
      this._gl.FLOAT,
      data,
      0
    );

    this._gl.drawArrays(this._gl.TRIANGLES, 0, this._verts.length / 2);
  }
}
