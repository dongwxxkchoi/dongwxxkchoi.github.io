/**
 * Menu
 */
document.addEventListener("DOMContentLoaded", function() {
  var menuIcon = document.querySelector("a.menu-icon");
  var menu = document.querySelector(".site-nav .menu");

  if (!menuIcon || !menu) {
    return;
  }

  menuIcon.addEventListener("click", function(event) {
    event.preventDefault();

    var isOpen = menu.style.display === "block";
    menu.style.display = isOpen ? "none" : "block";
    menuIcon.setAttribute("aria-expanded", isOpen ? "false" : "true");
  });
});

/**
 * Footer year
 */
$(document).ready(function() {
  $("#year").text(new Date().getFullYear());
});

/**
 * Footer email
 */
$(document).ready(function() {
  $(".email-link-cloaked").on("click", function(event) {
    const _link = event.target;

    const _user = reverseString(_link.getAttribute("data-resu"));
    const _domain = reverseString(_link.getAttribute("data-eman-niamod"));
    const _tld = reverseString(_link.getAttribute("data-dlt-niamod"));

    window.location.href = `mailto:${_user}@${_domain}.${_tld}`;
  });
});

/**
 * Reverses a string
 */
function reverseString(str) {
  return str.split("").reverse().join("");
}

/**
 * Moves WeChat widget
 */
function moveWidget(event) {
  var w = $("#wechat-widget");

  w.css({
    left: event.pageX - 25,
    top: event.pageY - w.height() - 60
  });
}

$("a#wechat-link").on("mouseenter", function(event) {
  $("#wechat-widget").css({ display: "block" });

  moveWidget(event);
});

$("a#wechat-link").on("mousemove", function(event) {
  moveWidget(event);
});

$("a#wechat-link").on("mouseleave", function(event) {
  $("#wechat-widget").css({ display: "none" });
});
