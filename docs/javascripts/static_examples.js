document$.subscribe(function() {
    // Force open all example boxes and keep them open
    document.querySelectorAll("details.example").forEach(function(detail) {
        detail.setAttribute("open", "");
    });
});
