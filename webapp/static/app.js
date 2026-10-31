/* BrainScan AI — shared frontend helpers */

const BrainScan = (() => {
  function token() {
    return localStorage.getItem("token");
  }

  function authHeaders(json = true) {
    const headers = { Authorization: "Bearer " + token() };
    if (json) headers["Content-Type"] = "application/json";
    return headers;
  }

  function logout() {
    localStorage.clear();
    window.location.href = "/login";
  }

  function statusLabel(status) {
    const map = {
      pending: "Pending review",
      approved: "Approved",
      rejected: "Rejected",
      further_review: "Further review",
      reviewed: "Reviewed",
      none: "None",
      verified: "Verified",
      suspended: "Suspended",
    };
    return map[status] || status || "—";
  }

  function badgeClass(status) {
    return "badge badge--" + (status || "none");
  }

  function formatPct(conf) {
    if (conf === null || conf === undefined) return "—";
    return (Number(conf) * 100).toFixed(1) + "%";
  }

  function formatDate(value) {
    if (!value) return "—";
    return new Date(value).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  function formatDateTime(value) {
    if (!value) return "—";
    return new Date(value).toLocaleString();
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function api(url, options = {}) {
    const res = await fetch(url, options);
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      data = null;
    }
    return { res, data };
  }

  async function requireAuth(allowedRoles) {
    if (!token()) {
      window.location.href = "/login";
      return null;
    }
    const { res, data } = await api("/api/auth/me", {
      headers: authHeaders(false),
    });
    if (!res.ok) {
      logout();
      return null;
    }
    localStorage.setItem("role", data.role);
    localStorage.setItem("name", data.name);
    localStorage.setItem("verification_status", data.verification_status || "none");

    if (allowedRoles && !allowedRoles.includes(data.role)) {
      alert("Access denied for this area.");
      logout();
      return null;
    }

    // Pending doctor applicants land on status page if they somehow hit doctor routes
    if (
      allowedRoles &&
      allowedRoles.includes("doctor") &&
      data.role === "doctor" &&
      data.verification_status !== "verified" &&
      !window.location.pathname.startsWith("/doctor/pending")
    ) {
      window.location.href = "/doctor/pending";
      return null;
    }

    if (
      data.role === "patient" &&
      data.verification_status === "pending" &&
      window.location.pathname.startsWith("/doctor")
    ) {
      window.location.href = "/doctor/pending";
      return null;
    }

    return data;
  }

  function wireSidebar() {
    const toggle = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebarBackdrop");
    if (!toggle || !sidebar) return;
    const close = () => {
      sidebar.classList.remove("is-open");
      backdrop && backdrop.classList.remove("is-open");
    };
    toggle.addEventListener("click", () => {
      sidebar.classList.toggle("is-open");
      backdrop && backdrop.classList.toggle("is-open");
    });
    backdrop && backdrop.addEventListener("click", close);
  }

  function fillUserChrome(user) {
    document.querySelectorAll("[data-user-name]").forEach((el) => {
      el.textContent = user.name || "";
    });
    document.querySelectorAll("[data-user-role]").forEach((el) => {
      el.textContent = (user.role || "").toUpperCase();
    });
  }

  async function downloadReport(scanId) {
    if (!token()) {
      window.location.href = "/login";
      return;
    }
    const res = await fetch("/api/patient/scan/" + scanId + "/report", {
      headers: { Authorization: "Bearer " + token() },
    });
    if (!res.ok) {
      let detail = "Unable to download report.";
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {}
      alert(detail);
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "BrainScan_Report_" + scanId + ".pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function redirectAfterLogin(role, verificationStatus) {
    if (role === "admin") {
      window.location.href = "/admin/dashboard";
      return;
    }
    if (role === "doctor") {
      window.location.href =
        verificationStatus === "verified" ? "/doctor/dashboard" : "/doctor/pending";
      return;
    }
    if (verificationStatus === "pending") {
      // Patient with pending doctor application — patient dashboard + notice
      window.location.href = "/patient/dashboard";
      return;
    }
    window.location.href = "/patient/dashboard";
  }

  return {
    token,
    authHeaders,
    logout,
    statusLabel,
    badgeClass,
    formatPct,
    formatDate,
    formatDateTime,
    escapeHtml,
    api,
    requireAuth,
    wireSidebar,
    fillUserChrome,
    downloadReport,
    redirectAfterLogin,
  };
})();

window.BrainScan = BrainScan;
window.logout = BrainScan.logout;
window.downloadReport = BrainScan.downloadReport;
