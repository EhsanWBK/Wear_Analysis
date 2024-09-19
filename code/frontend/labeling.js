"use strict";

const elementsInput = document.getElementsByTagName('input');

function py_changeMaskExtension() {
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
	eel.change_mask_extension(parameters)();
}