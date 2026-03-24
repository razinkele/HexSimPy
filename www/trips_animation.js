// TripsLayer animation — injects into the EXISTING shiny_deckgl overlay.
// Uses deck.gl layer diffing: passes existing layer instances unchanged
// plus a new TripsLayer. deck.gl only updates what changed.

(function () {
  if (window._tripsAnimRegistered) return;
  window._tripsAnimRegistered = true;

  var _rafId = null;
  var _tripsData = null;
  var _animSpeed = 3;
  var _trailLength = 10;
  var _loopLength = 24;
  var _currentTime = 0;
  var _targetId = 'map';
  var _lastDeckLayers = null;  // cache of non-trips deck layers

  function waitForShiny() {
    if (typeof Shiny !== 'undefined' && Shiny.addCustomMessageHandler) {
      register();
    } else {
      setTimeout(waitForShiny, 200);
    }
  }

  function register() {
    Shiny.addCustomMessageHandler('deck_trips_data', function (payload) {
      _tripsData = payload.data;
      _loopLength = payload.loopLength || 24;
      _trailLength = payload.trailLength || 10;
      _animSpeed = payload.speed || 3;
      _currentTime = 0;
      _lastDeckLayers = null;  // force refresh
      if (_tripsData && _tripsData.length > 0) startAnimation();
      else stopAnimation();
    });
    Shiny.addCustomMessageHandler('deck_trips_stop', function () {
      stopAnimation();
    });
  }

  function getInstance() {
    return window.__deckgl_instances && window.__deckgl_instances[_targetId];
  }

  function startAnimation() {
    if (_rafId) cancelAnimationFrame(_rafId);
    _rafId = requestAnimationFrame(tick);
  }

  function stopAnimation() {
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    _tripsData = null;
    _lastDeckLayers = null;
    // Remove trips layer by re-rendering without it
    var inst = getInstance();
    if (inst && inst.overlay && inst.overlay._deck) {
      var mgr = inst.overlay._deck.layerManager;
      if (mgr) {
        var layers = mgr.getLayers().filter(function(l) { return l.id !== 'anim-trails'; });
        inst.overlay.setProps({ layers: layers });
        inst.map.triggerRepaint();
      }
    }
  }

  function tick() {
    _currentTime = (_currentTime + _animSpeed * 0.016) % _loopLength;

    var inst = getInstance();
    if (!inst || !inst.overlay || !inst.overlay._deck) {
      _rafId = requestAnimationFrame(tick);
      return;
    }

    // Get current layers from the layer manager (instantiated objects)
    var mgr = inst.overlay._deck.layerManager;
    var existing = mgr.getLayers().filter(function(l) { return l.id !== 'anim-trails'; });

    // Create fresh TripsLayer with updated currentTime
    var tripsLayer = new deck.TripsLayer({
      id: 'anim-trails',
      data: _tripsData,
      getPath: function (d) { return d.path; },
      getTimestamps: function (d) { return d.timestamps; },
      getColor: function (d) { return d.color; },
      widthMinPixels: 5,
      widthMaxPixels: 10,
      jointRounded: true,
      capRounded: true,
      trailLength: _trailLength,
      currentTime: _currentTime,
    });

    // Insert trips between water (idx 0) and agents (idx 1+)
    var waterIdx = -1;
    for (var i = 0; i < existing.length; i++) {
      if (existing[i].id === 'water') { waterIdx = i; break; }
    }
    if (waterIdx >= 0) {
      existing.splice(waterIdx + 1, 0, tripsLayer);
    } else {
      existing.unshift(tripsLayer);
    }
    inst.overlay.setProps({ layers: existing });
    inst.map.triggerRepaint();

    _rafId = requestAnimationFrame(tick);
  }

  waitForShiny();
})();
