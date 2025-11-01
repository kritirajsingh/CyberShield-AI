document.addEventListener("DOMContentLoaded", () => {
  // ---------- FORM HANDLING ----------
  const forms = document.querySelectorAll("form");
  forms.forEach(form => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const flashMessage = document.querySelector(".flash");
      const resultsDiv = document.querySelector(".results");
      flashMessage.textContent = "";
      resultsDiv.innerHTML = "🔍 Scanning... Please wait.";

      const formData = new FormData(form);
      const data = Object.fromEntries(formData);

      try {
        const response = await fetch(form.action, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data)
        });

        if (!response.ok) {
          const err = await response.json();
          flashMessage.textContent = err.error || "Invalid input!";
          flashMessage.classList.add("flash", "danger");
          resultsDiv.innerHTML = "";
          return;
        }

        const result = await response.text();
        resultsDiv.innerHTML = `<p>${result}</p>`;
        flashMessage.classList.add("flash", "success");
      } catch (err) {
        flashMessage.textContent = "⚠️ Error connecting to server.";
        flashMessage.classList.add("flash", "danger");
        resultsDiv.innerHTML = "";
      }
    });
  });

  // ---------- AI Password Strength (frontend) ----------
  const aiBtn = document.querySelector("#checkPasswordBtn");
  if (aiBtn) {
    aiBtn.addEventListener("click", async () => {
      const passwordInput = document.querySelector("#passwordInput");
      const feedbackDiv = document.querySelector(".ai-feedback");
      const flash = document.querySelector(".flash");

      if (!passwordInput.value.trim()) {
        flash.textContent = "Please enter a password!";
        flash.classList.add("flash", "info");
        return;
      }

      feedbackDiv.innerHTML = "🤖 Analyzing password using AI...";

      try {
        const response = await fetch("/ai_password_check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password: passwordInput.value })
        });

        const data = await response.json();
        if (response.ok) {
          feedbackDiv.innerHTML = `
            <p><b>Strength:</b> ${data.Strength}</p>
            <p>${data.AI_Feedback}</p>
          `;
        } else {
          feedbackDiv.textContent = "❌ Error: " + data.error;
        }
      } catch (err) {
        feedbackDiv.textContent = "⚠️ Could not connect to AI service.";
      }
    });
  }

  // ---------- Theme Toggle ----------
  const themeToggle = document.querySelector(".theme-btn");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
      localStorage.setItem("theme", document.body.classList.contains("dark-mode") ? "dark" : "light");
    });

    // Restore saved theme
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") document.body.classList.add("dark-mode");
  }
});
