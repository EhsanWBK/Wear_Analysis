"use strict";

const canvas = document.getElementById('orginalCanvas');
const resetButton = document.getElementById('resetButton');
const slider = document.getElementById('transparenzSlider');
const outputSlider = document.getElementById('transparenzOutput');
const outputCanvasPoints = document.getElementById('numPointsOutput');
const pointDistances = document.getElementById('distance');
const buttonEnableCam = document.getElementById('enableCam');
const buttonTakePicture = document.getElementById('takePicture');
buttonTakePicture.disabled = true;
const imgAufnahme = document.getElementById('aufnahme');
const imgStream = document.getElementById('bg');
const imgMask = document.getElementById('maskPicture');
/* const radioButtons = document.querySelector('[input[name="zoom"]'); */
const dropdownMenu = document.getElementById('dropDown');
const buttonSegment = document.getElementById('idbuttonSegment');
buttonSegment.disabled = true;
const imageFileInput = document.getElementById('idimageFileInput');
const loadModelInput = document.getElementById('idloadModelInput');
const cameraInput = document.getElementById('idCameraInput');
const buttonReshape = document.getElementById('idButtonReshape');
buttonReshape.disabled = true;


const inputPictureMaskDir = document.getElementById('SavePictureDirectory_str');

let ctx; 
let mouseX, mouseY, mouseDown = 0;
let touchX, touchY;
let x1, y1, x2, y2, x3, y3, x4, y4, firstX, firstY, lastX, lastY;
var dist1, dist2, dist3, dist4;
let numPoints;
numPoints = 0;
/* alert("Konstanten geladen");	 */
				

function drawDot(ctx,x,y,size) {
	const r=0; 
	const g=0; 
	const b=0; 
	const a=255;
	ctx.fillStyle = "#74bdf7";

	ctx.beginPath();
	ctx.arc(x, y, size, 0, Math.PI*2, true); 
	ctx.closePath();
	ctx.fill();
	
	
	numPoints = numPoints + 1;
	outputCanvasPoints.value = numPoints;

	if (numPoints == 1){
		firstX = x;
		firstY =y;
	}
	if (numPoints > 1) {	
		
		
		//Linie zeichnen
		ctx.strokeStyle = 'red';
		ctx.setwidth = 20;
		ctx.beginPath();
		ctx.moveTo(lastX, lastY);
		ctx.lineTo(x, y);
		if (numPoints==4){
			ctx.moveTo(x,y)
			ctx.lineTo(firstX, firstY)
		}
		/* ctx.closePath(); */
		ctx.stroke();	
		
		
	}
	if (numPoints == 4) {
		/* ctx.stroke(); */
		//Interaktivitaet beenden

		canvas.removeEventListener('mousedown', sketch_mouseDown);
		canvas.removeEventListener('mousemove', sketch_mouseMove);
		window.removeEventListener('mouseup', sketch_mouseUp);
	}
	lastX = x;
	lastY = y;
	
	//Punkte fuer Funktionen entsprechend der Skalierung anpassen
	let x_scaled = x / imgAufnahme.width * imgAufnahme.naturalWidth;
	let y_scaled = y / imgAufnahme.height * imgAufnahme.naturalHeight;
	
	switch (numPoints) {
		case 1:
			x1 = x_scaled;
			y1 = y_scaled;
		case 2:
			x2 = x_scaled;
			y2 = y_scaled;
		case 3:
			x3 = x_scaled;
			y3 = y_scaled;
		case 4:
			x4 = x_scaled;
			y4 = y_scaled;
	}
	
	
}

function calcDistance(){
	var dist = 0;
	switch (numPoints){
		case 1:
			dist = x1;
		case 2:
			a2 = x2-x1;
			b2 = y2-y1;
			dist2 = Math.sqrt(a2*a2+b2*b2);
			pointDistances.value = dist2;
			//alert('x1:'+x1+'\ty1:'+y1+'\nx2:'+x2+'\ty2:'+y2);
		case 3:
			a3 = x3-x1;
			b3 = y3-y1;
			dist3 = Math.sqrt(a3*a3+b3*b3);
			pointDistances.value = dist3;
			//dist = Math.abs((x3-x1)+(y3-y1));
		case 4:
			a4 = x4-x3;
			b4 = y4-y3;
			dist4 = Math.sqrt(a4*a4+b4*b4);
			pointDistances.value = dist4;
			//dist = Math.abs((x4-x3)+(y4-y3));

	}
	pointDistances.value = dist;
}


/* function clearCanvas(canvas,ctx) { */
function clearCanvas() {
	/* alert("Canvas soll gereinigt werden"); */
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	/* alert("Canvas cleared"); */
	
	//Reinitialisiere Funktionen
	if (numPoints == 4) {
		canvas.addEventListener('mousedown', sketch_mouseDown);
		canvas.addEventListener('mousemove', sketch_mouseMove);
		window.addEventListener('mouseup', sketch_mouseUp);
	}
	numPoints = 0;
	outputCanvasPoints.value = numPoints;
	pointDistances.value=numPoints;
	mouseDown = 0;
	x1,y1,x2,y2,x3,y3,x4,y4=null;
}

function sketch_mouseDown() {
	mouseDown = 1;
	drawDot(ctx,mouseX,mouseY,6);
	calcDistance();
	
	 
	
}

function sketch_mouseUp() {
	mouseDown = 0;
}

function sketch_mouseMove(e) { 
	getMousePos(e);
	/*
	if (mouseDown==1) {
		drawDot(ctx,mouseX,mouseY,6);
	}*/
}

function getMousePos(e) {
	if (!e) {
		let e = event;
	}

	if (e.offsetX) {
		mouseX = e.offsetX;
		mouseY = e.offsetY;
	}
	else if (e.layerX) {
		mouseX = e.layerX;
		mouseY = e.layerY;
	}
 }
 
 function isInputEmpty(element,indey,array){
	 return element == null || element =="";
 }
 

 
function py_video() {
	/*eel.video_feed(cameraInput.value)()*/
	eel.videoFeed()()
	buttonTakePicture.disabled = !buttonTakePicture.disabled;
}

function py_foto() {
	/* alert("take picture"); */
	eel.takePicture()()
	
}

function py_segment() {
	/*alert("Segmentation"); */
	eel.segmentImage()();
}

function py_getPos(){
	eel.getPos(x1,x2,y1,y2);
}

function py_calibrate() {
	let distance = 0.0;
	if (outputCanvasPoints.value == 2) {
		distance = Math.sqrt((x1-x2)**2 + (y1-y2)**2);
		eel.calibrateDistance(distance)();
		py_getPos();
		dropdownMenu.value = distance;
		
	}
	else {
		distance = dropdownMenu.value;
		eel.calibrateDistance(distance)();
	}
	
	
}



function py_loadImage() {
	let path = imageFileInput.value;
	if (path == null || path ==""){
		alert("Input field empty");
	}
	else {
		/* alert(path); */
		eel.loadImage(path)();
	}
}

function py_loadModel() {
	let model = loadModelInput.value
	if (model == null || model ==""){
		alert("Input field empty");
	}
	else {
		eel.loadModel(model)();
		buttonSegment.disabled = false;
	}
}

function py_initReshape() {
	if (numPoints < 4) {
		alert("Not enough points marked.");
	}
	else {
		eel.init_reshape(x1,y1,x2,y2,x3,y3,x4,y4)();
		buttonReshape.disabled = false;
	}
}

function py_reshape() {
	eel.reshape_image()();
}

function py_autoCrop() {
	eel.auto_crop()();
}

function py_savePictureMask() {
	console.log("saving picture and mask at: "+ inputPictureMaskDir.value);
	eel.saveImage(inputPictureMaskDir.value)();
}

function py_try() {
	eel.tryOut()();
}

eel.expose(updateImageSrc);
function updateImageSrc(val) {
	
	imgStream.src = "data:image/jpeg;base64," + val
}



eel.expose(updatePicture);
function updatePicture(val) {
	imgAufnahme.src = "data:image/jpeg;base64," + val
	imgMask.src = '#';
	//clearCanvas();
}

eel.expose(updateMask);
function updateMask(val) {
	imgMask.src = "data:image/jpeg;base64," + val
}

eel.expose(addDropdown);
function addDropdown(name){
	/* alert(name); */
	name = Math.round(name * 100) / 100;
	var option = document.createElement('option');
	/*option.text=name;*/
	option.value=name;
	option.innerHTML = name;
	/*dropdownMenu.add(option);*/
	dropdownMenu.appendChild(option);
}

function init() {

	if (canvas.getContext)
		ctx = canvas.getContext('2d');
		/* var background = new Image();
		background.src = "img/Validaion15.jpg";
		// Make sure the image is loaded first otherwise nothing will draw.
		background.onload = function(){
			ctx.drawImage(background,0,0);   
		} */

	// Draw whatever else over top of it on the canvas.
	if (ctx) {
		canvas.addEventListener('mousedown', sketch_mouseDown);
		canvas.addEventListener('mousemove', sketch_mouseMove);
		window.addEventListener('mouseup', sketch_mouseUp);
		/* resetButton.addEventListener("click", function() {
		clearCanvas(canvas, ctx);}); */
		resetButton.addEventListener("click", clearCanvas);
		document.addEventListener('DOMContentLoaded', function() {
			outputSlider.value = slider.value;
		});

		slider.addEventListener ("input", function () {
			outputSlider.value = this.value;
			imgMask.style.opacity = this.value / 100;
		});
		/* buttonEnableCam.addEventListener('click', py_video()); */
		/*fill Options in Dropdown-Menu*/
		/*eel.readCalibrationParameter()()*/
		
		


	}
}

init();

			// Bild über HTML Formular mit einem hidden field (save_remote_data) an Server senden.
			// canvas-to-png.php ist ein PHP-Script auf dem Server
			// Quelle: http://stackoverflow.com/questions/13198131/how-to-save-a-html5-canvas-as-image-on-a-server

			/* document.querySelector("#savepng").addEventListener("submit", function () {
				var image_data = canvas.toDataURL("image/png");
				document.getElementById('save_remote_data').value = image_data; // Place the image data in to the form
			}); */