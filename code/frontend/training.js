"use strict";

// Get elements
const elementsInput = document.getElementsByTagName('input');
console.log(elementsInput);

const buttonTensorboard = document.getElementById('button2tensorboard');
/* buttonTensorboard.disabled = true; */
const buttonMode = document.getElementById('idMode');
const elementsExpert = document.getElementsByClassName('expertMode');



function change_mode() {
	if (elementsExpert[0].style.display == 'none') {
		for(let i=0; i<elementsExpert.length; i++) {
			elementsExpert[i].style.display = 'initial';
		}
		buttonMode.innerHTML = 'to Normal Mode';
	}
	else {		
		for(let i=0; i<elementsExpert.length; i++) {
			elementsExpert[i].style.display = 'none';
		}
		buttonMode.innerHTML = 'to Expert Mode';
	}
	
}



function py_train() {
	/* buttonTensorboard.disabled = false; */
	var parameters = [];
	parameters.length = elementsInput.length;
	var p = [];
	p.length = 2;
	for(let i=0; i<elementsInput.length; i++) {
		p[0] = elementsInput[i].id;
		console.log(p[0]);
		switch (elementsInput[i].type) {
			case 'checkbox':
				p[1] = elementsInput[i].checked;
				break;
			case 'number':
				p[1] = elementsInput[i].value;
				break;
			default:
				p[1] = elementsInput[i].value;
		}
		console.log(p[1]);
		parameters[i]=p.slice();
	}
	console.table(parameters);
	eel.trainModel(parameters)();
}

function py_launchTensorboard() {
	buttonTensorboard.href = eel.launchTensorboard()();
}
	
eel.expose(gotoTensorboard);
function gotoTensorboard(url) {
	window.open(url,"_blank").focus();
}

//Actions
change_mode();