// JavaScript to update file name display
document.getElementById('fileInput1').addEventListener('change', function () {
    const fileName1 = document.getElementById('fileName1');
    fileName1.textContent = this.files[0] ? this.files[0].name : 'No file selected';
});

document.getElementById('fileInput2').addEventListener('change', function () {
    const fileName2 = document.getElementById('fileName2');
    fileName2.textContent = this.files[0] ? this.files[0].name : 'No file selected';
});