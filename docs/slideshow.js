var slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides(slideIndex += n);
  clearTimeout(timeoutHandle);
  myLoop()
}

function currentSlide(n) {
  showSlides(slideIndex = n);
  clearTimeout(timeoutHandle);
  myLoop()
}

function showSlides(n) {
  var i;
  var slides = document.getElementsByClassName("mySlides");
  // var dots = document.getElementsByClassName("dot");
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";
  }
  // for (i = 0; i < dots.length; i++) {
  //     dots[i].className = dots[i].className.replace(" active", "");
  // }
  if (slideIndex > slides.length) {slideIndex = 1}
  slides[slideIndex-1].style.display = "block";
  // dots[slideIndex-1].className += " active";
}

var i = 1;
function myLoop() {
  timeoutHandle = setTimeout(function() {
    showSlides(slideIndex += 1);
    i++;
    if (i < 100) {myLoop();}
  }, 4500)
}

myLoop();
