// JavaScript to update file name display
document.querySelector('#fileInput').addEventListener('change', function() {
    const fileName = document.querySelector('#fileName');
    fileName.textContent = this.files[0] ? this.files[0].name : 'No file selected';
});