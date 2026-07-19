// ── DOM References ───────────────────────────────────────────────────────────
const engineSelect = document.getElementById('engine-select');
const modeStatic = document.getElementById('mode-static');
const modeSimulate = document.getElementById('mode-simulate');
const simControls = document.getElementById('simulation-controls');
const simPlayPause = document.getElementById('sim-play-pause');
const simReset = document.getElementById('sim-reset');
const simSpeed = document.getElementById('sim-speed');
const speedVal = document.getElementById('speed-val');
const playIcon = document.getElementById('play-icon');
const playText = document.getElementById('play-text');

const healthBadge = document.getElementById('health-badge');
const healthStatusText = document.getElementById('health-status-text');
const metricCycle = document.getElementById('metric-cycle');
const metricTotalCycles = document.getElementById('metric-total-cycles');
const metricRfProb = document.getElementById('metric-rf-prob');
const metricTrueRul = document.getElementById('metric-true-rul');
const rfProgress = document.getElementById('rf-progress');

const llmVerdictCard = document.getElementById('llm-verdict-card');
const llmStatusTag = document.getElementById('llm-status-tag');
const llmReason = document.getElementById('llm-reason');
const llmConfFill = document.getElementById('llm-conf-fill');
const llmConfVal = document.getElementById('llm-conf-val');

// ── State ────────────────────────────────────────────────────────────────────
let currentEngineId = null;
let engineData = null;
let activeMode = 'static';
let charts = {};
let isPlaying = false;
let simCycle = 1;
let simTimer = null;
let simIntervalMs = 1000;

// Sensors shown in the UI (the 6 primary raw sensors)
const sensorList = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11'];

// ── Chart Initialization ─────────────────────────────────────────────────────
function initCharts() {
    sensorList.forEach(sensor => {
        const canvas = document.getElementById(`chart-${sensor.replace('_', '-')}`);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        charts[sensor] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: sensor,
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.08)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 200 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#18181b',
                        borderColor: '#27272a',
                        borderWidth: 1,
                        titleColor: '#a1a1aa',
                        bodyColor: '#fafafa',
                        bodyFont: { family: 'JetBrains Mono', size: 11 },
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#52525b', font: { size: 10 }, maxTicksLimit: 8 },
                        grid: { color: 'rgba(39, 39, 42, 0.5)' },
                    },
                    y: {
                        min: 0,
                        max: 1,
                        ticks: { color: '#52525b', font: { size: 10 }, stepSize: 0.25 },
                        grid: { color: 'rgba(39, 39, 42, 0.5)' },
                    }
                }
            }
        });
    });
}

// ── Engine Loading ───────────────────────────────────────────────────────────
async function fetchEngines() {
    try {
        const res = await fetch('/api/engines');
        const data = await res.json();
        
        engineSelect.innerHTML = '';
        data.engines.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = `Engine #${id}`;
            engineSelect.appendChild(opt);
        });
        
        currentEngineId = data.engines[0];
        loadEngine(currentEngineId);
    } catch (err) {
        console.error('Failed to fetch engines:', err);
    }
}

async function loadEngine(engineId) {
    stopSimulation();
    simCycle = 1;
    engineData = null;

    try {
        const res = await fetch(`/api/engine/${engineId}`);
        engineData = await res.json();
        
        metricTotalCycles.textContent = engineData.max_cycle;
        
        if (activeMode === 'static') {
            runStaticAnalysis();
        } else {
            resetSimulation();
        }
    } catch (err) {
        console.error('Failed to load engine data:', err);
    }
}

// ── Verdict UI Update ────────────────────────────────────────────────────────
function updateVerdictUI(verdict) {
    const finalLabel = verdict.final_label || 'Healthy';
    
    // Status badge
    healthStatusText.textContent = finalLabel.toUpperCase();
    healthBadge.className = 'badge';
    
    if (finalLabel === 'Critical') {
        healthBadge.classList.add('badge-critical');
    } else if (finalLabel === 'Warning') {
        healthBadge.classList.add('badge-warning');
    } else {
        healthBadge.classList.add('badge-healthy');
    }
    
    // RF probability bar
    metricRfProb.textContent = verdict.rf_prob_bad.toFixed(2);
    rfProgress.style.width = `${verdict.rf_prob_bad * 100}%`;
    
    if (verdict.rf_prob_bad > 0.8) {
        rfProgress.style.background = 'var(--color-critical)';
    } else if (verdict.rf_prob_bad > 0.5) {
        rfProgress.style.background = 'var(--color-warning)';
    } else {
        rfProgress.style.background = 'linear-gradient(to right, #6366f1, #a78bfa)';
    }
    
    // LLM verdict panel
    if (verdict.llm_response && verdict.llm_response.status !== 'LLM_UNAVAILABLE') {
        const llm = verdict.llm_response;
        llmStatusTag.textContent = llm.status.toUpperCase();
        llmStatusTag.className = 'llm-status-tag';
        llmStatusTag.classList.add(llm.status.toLowerCase());
        
        llmReason.textContent = `"${llm.reason}"`;
        
        const confPercent = Math.round(llm.confidence * 100);
        llmConfVal.textContent = `${confPercent}%`;
        llmConfFill.style.width = `${confPercent}%`;
        llmVerdictCard.style.borderColor = llm.status === 'Bad' ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)';
    } else {
        llmStatusTag.textContent = 'STANDBY';
        llmStatusTag.className = 'llm-status-tag';
        
        if (activeMode === 'simulate') {
            llmReason.textContent = 'LLM paused during simulation to conserve API tokens.';
        } else {
            llmReason.textContent = 'Select static mode to generate LLM diagnostic verdict.';
        }
        
        llmConfVal.textContent = '0%';
        llmConfFill.style.width = '0%';
        llmVerdictCard.style.borderColor = 'var(--border-color)';
    }
}

// ── Static Analysis Mode ─────────────────────────────────────────────────────
async function runStaticAnalysis() {
    if (!engineData) return;
    
    const maxCycle = engineData.max_cycle;
    metricCycle.textContent = maxCycle;
    
    // Plot full trajectory
    const labels = engineData.data.map(r => r.time_in_cycles);
    sensorList.forEach(sensor => {
        const valData = engineData.data.map(r => r[sensor]);
        charts[sensor].data.labels = labels;
        charts[sensor].data.datasets[0].data = valData;
        charts[sensor].options.animation.duration = 300;
        charts[sensor].update();
    });
    
    // LLM loading state
    llmStatusTag.textContent = 'THINKING...';
    llmReason.textContent = 'AeroShield LLM is analyzing current trajectory telemetry...';
    llmConfFill.style.width = '20%';
    
    try {
        const res = await fetch(`/api/predict?engine_id=${currentEngineId}&cycle=${maxCycle}&run_llm=true`);
        const verdict = await res.json();
        
        metricTrueRul.textContent = engineData.true_rul;
        updateVerdictUI(verdict);
    } catch (err) {
        console.error('Error fetching static prediction:', err);
        llmStatusTag.textContent = 'ERROR';
        llmReason.textContent = 'Failed to retrieve LLM second opinion.';
    }
}

// ── Simulation Streaming Mode ────────────────────────────────────────────────
function resetSimulation() {
    stopSimulation();
    simCycle = 1;
    metricCycle.textContent = 0;
    metricTrueRul.textContent = 'N/A';
    
    sensorList.forEach(sensor => {
        charts[sensor].data.labels = [];
        charts[sensor].data.datasets[0].data = [];
        charts[sensor].options.animation.duration = 0;
        charts[sensor].update();
    });
    
    updateVerdictUI({
        rf_prob_bad: 0.0,
        final_label: 'Healthy',
        color: 'green',
        llm_response: { status: 'LLM_UNAVAILABLE', confidence: 0, reason: '' }
    });
}

function startSimulation() {
    if (isPlaying) return;
    isPlaying = true;
    playIcon.textContent = '\u23F8';
    playText.textContent = 'Pause';
    
    runSimTick();
}

function stopSimulation() {
    isPlaying = false;
    if (simTimer) {
        clearTimeout(simTimer);
        simTimer = null;
    }
    playIcon.textContent = '\u25B6';
    playText.textContent = 'Resume';
}

async function runSimTick() {
    if (!isPlaying || !engineData) return;
    
    if (simCycle > engineData.max_cycle) {
        stopSimulation();
        return;
    }
    
    metricCycle.textContent = simCycle;
    
    const remainingTrueRul = engineData.true_rul + (engineData.max_cycle - simCycle);
    metricTrueRul.textContent = remainingTrueRul;
    
    const row = engineData.data[simCycle - 1];
    sensorList.forEach(sensor => {
        charts[sensor].data.labels.push(row.time_in_cycles);
        charts[sensor].data.datasets[0].data.push(row[sensor]);
        
        // Keep rolling 100-cycle window
        if (charts[sensor].data.labels.length > 100) {
            charts[sensor].data.labels.shift();
            charts[sensor].data.datasets[0].data.shift();
        }
        charts[sensor].update();
    });
    
    try {
        const res = await fetch(`/api/predict?engine_id=${currentEngineId}&cycle=${simCycle}&run_llm=false`);
        const verdict = await res.json();
        updateVerdictUI(verdict);
    } catch (err) {
        console.error('Error during simulation tick:', err);
    }
    
    simCycle++;
    simTimer = setTimeout(runSimTick, simIntervalMs);
}

// ── Event Listeners ──────────────────────────────────────────────────────────

engineSelect.addEventListener('change', (e) => {
    currentEngineId = parseInt(e.target.value);
    loadEngine(currentEngineId);
});

modeStatic.addEventListener('click', () => {
    if (activeMode === 'static') return;
    activeMode = 'static';
    modeStatic.classList.add('active');
    modeSimulate.classList.remove('active');
    simControls.classList.add('hidden');
    loadEngine(currentEngineId);
});

modeSimulate.addEventListener('click', () => {
    if (activeMode === 'simulate') return;
    activeMode = 'simulate';
    modeSimulate.classList.add('active');
    modeStatic.classList.remove('active');
    simControls.classList.remove('hidden');
    loadEngine(currentEngineId);
});

simPlayPause.addEventListener('click', () => {
    if (isPlaying) {
        stopSimulation();
    } else {
        startSimulation();
    }
});

simReset.addEventListener('click', () => {
    resetSimulation();
});

simSpeed.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    simIntervalMs = value;
    speedVal.textContent = `${(value / 1000).toFixed(1)}s`;
});

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchEngines();
});
