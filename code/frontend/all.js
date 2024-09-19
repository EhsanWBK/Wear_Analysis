"use strict";

/* const textConnection = document.getElementById('idConnection');
let connected = false; */

function py_directory(elementId) {
	eel.getDirectory(elementId)();	
}

function py_file(elementId) {
	eel.getFile(elementId)();	
}

/* function connection_lost() {
	if (!connected) {
		textConnection.innerHTML = "pyton not connected";
		textConnection.style.color = "red";
	}
	else {
		textConnection.innerHTML = "pyton connected";
		textConnection.style.color = "green";
		connected = false;
	}
	
}
 */


eel.expose(updateDirectoryName);
function updateDirectoryName(directory, elementId) {
	const inputElement = document.getElementById(elementId);
	//alert(inputElement);
	inputElement.value = directory;
}

/* eel.expose(connectionCheck)
function connectionCheck() {
	connected = true;
}

let timer_interval = window.setInterval(connection_lost, 10000); */

	
	