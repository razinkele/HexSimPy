/* Streaming charts — receives JSON pushes from Python, updates Plotly traces.
 * No re-renders, no iframes, no disk I/O. O(1) append per step.
 */
(function() {
  "use strict";

  var MAX_POINTS = 200;
  var initialized = false;

  var COLORS = {
    alive:      "#6bcb77",
    dead:       "#ff6b6b",
    arrived:    "#ffd93d",
    hold:       "#7a8b7a",
    random:     "#4a8fa8",
    cwr:        "#3d9b8f",
    upstream:   "#d4826a",
    downstream: "#b8963e",
    migration:  "#ffd93d",
  };

  var CHART_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { color: "#aaa", size: 10 },
    margin: { l: 35, r: 10, t: 5, b: 25 },
    showlegend: false,
    xaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
    yaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
  };

  function initPopulation(msg) {
    if (typeof Plotly === 'undefined') return;
    var el = document.getElementById("chart-population");
    if (!el) return;
    var traces = [
      { x: [], y: [], mode: "lines", name: "Alive",
        line: { color: COLORS.alive, width: 2 }, fill: "tozeroy",
        fillcolor: "rgba(107,203,119,0.15)" },
      { x: [], y: [], mode: "lines", name: "Dead",
        line: { color: COLORS.dead, width: 1.5, dash: "dash" } },
      { x: [], y: [], mode: "lines", name: "Arrived",
        line: { color: COLORS.arrived, width: 1.5, dash: "dot" } },
    ];
    var layout = Object.assign({}, CHART_LAYOUT, {
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
        range: [0, msg.n_agents * 1.05],
      }),
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Hour", font: { size: 9 } },
      }),
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  function initMigration(msg) {
    if (typeof Plotly === 'undefined') return;
    var el = document.getElementById("chart-migration");
    if (!el) return;
    var nBins = msg.n_bins || 50;
    var edges = msg.bin_edges || [];
    var centers = [];
    for (var i = 0; i < edges.length - 1; i++) {
      centers.push((edges[i] + edges[i + 1]) / 2);
    }
    el._binCenters = centers;
    var traces = [{
      x: centers, y: new Array(centers.length).fill(0),
      type: "bar", marker: { color: COLORS.migration, opacity: 0.8 },
    }];
    var layout = Object.assign({}, CHART_LAYOUT, {
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Upstream (km)", font: { size: 9 } },
      }),
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
      }),
      bargap: 0.05,
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  function initBehavior(msg) {
    if (typeof Plotly === 'undefined') return;
    var el = document.getElementById("chart-behavior");
    if (!el) return;
    var names  = ["Hold", "Random", "CWR", "Upstream", "Downstream"];
    var colors = [COLORS.hold, COLORS.random, COLORS.cwr, COLORS.upstream, COLORS.downstream];
    var traces = names.map(function(name, i) {
      return {
        x: [], y: [], mode: "lines", name: name,
        stackgroup: "one", line: { color: colors[i], width: 0.5 },
        fillcolor: colors[i],
      };
    });
    var layout = Object.assign({}, CHART_LAYOUT, {
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
      }),
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Hour", font: { size: 9 } },
      }),
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  function updatePopulation(msg) {
    var el = document.getElementById("chart-population");
    if (!el || !el.data) return;
    Plotly.extendTraces(el,
      { x: [[msg.t], [msg.t], [msg.t]], y: [[msg.alive], [msg.dead], [msg.arrived]] },
      [0, 1, 2], MAX_POINTS);
  }

  function updateMigration(msg) {
    var el = document.getElementById("chart-migration");
    if (!el || !el.data || !el._binCenters) return;
    var bins = msg.migration_bins || [];
    Plotly.react(el, [{
      x: el._binCenters, y: bins,
      type: "bar", marker: { color: COLORS.migration, opacity: 0.8 },
    }], el.layout, { displayModeBar: false, responsive: true });
  }

  function updateBehavior(msg) {
    var el = document.getElementById("chart-behavior");
    if (!el || !el.data) return;
    var b = msg.behaviors;
    Plotly.extendTraces(el,
      { x: [[msg.t], [msg.t], [msg.t], [msg.t], [msg.t]],
        y: [[b.upstream], [b.hold], [b.random], [b.cwr], [b.downstream]] },
      [0, 1, 2, 3, 4], MAX_POINTS);
  }

  function setupPanel() {
    var handle = document.getElementById("charts-panel-handle");
    var body = document.getElementById("charts-panel-body");
    if (!handle || !body) return;

    handle.addEventListener("click", function() {
      var collapsed = body.style.display === "none";
      body.style.display = collapsed ? "flex" : "none";
      handle.querySelector("span").textContent = collapsed
        ? "\u25b2 LIVE CHARTS \u25b2"
        : "\u25bc LIVE CHARTS \u25bc";
      if (collapsed) {
        setTimeout(function() {
          ["chart-population", "chart-migration", "chart-behavior"].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) Plotly.Plots.resize(el);
          });
        }, 50);
      }
    });
  }

  function _waitForShiny() {
    if (window.Shiny && Shiny.addCustomMessageHandler) {
      Shiny.addCustomMessageHandler("chart_reset", function(msg) {
        try {
          initPopulation(msg);
          initMigration(msg);
          initBehavior(msg);
          initialized = true;
        } catch(e) {
          console.warn('[charts] reset failed:', e);
        }
      });

      Shiny.addCustomMessageHandler("chart_update", function(msg) {
        try {
          if (!initialized) return;
          updatePopulation(msg);
          updateMigration(msg);
          updateBehavior(msg);
        } catch(e) {
          console.warn('[charts] update failed:', e);
        }
      });
    } else {
      setTimeout(_waitForShiny, 200);
    }
  }
  _waitForShiny();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupPanel);
  } else {
    setupPanel();
  }

})();
