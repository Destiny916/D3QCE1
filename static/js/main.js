const canvas = document.getElementById("particle-canvas");

if (canvas) {
  const ctx = canvas.getContext("2d");
  const particles = [];
  const density = 70;
  const maxVelocity = 0.35;

  const resize = () => {
    const { innerWidth, innerHeight, devicePixelRatio } = window;
    canvas.width = innerWidth * devicePixelRatio;
    canvas.height = innerHeight * devicePixelRatio;
    canvas.style.width = `${innerWidth}px`;
    canvas.style.height = `${innerHeight}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(devicePixelRatio, devicePixelRatio);
  };

  const spawn = () => {
    particles.length = 0;
    for (let i = 0; i < density; i += 1) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * maxVelocity,
        vy: (Math.random() - 0.5) * maxVelocity,
        radius: 1.5 + Math.random() * 2.2,
        alpha: 0.2 + Math.random() * 0.5,
      });
    }
  };

  const draw = () => {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < -20 || p.x > window.innerWidth + 20) p.vx *= -1;
      if (p.y < -20 || p.y > window.innerHeight + 20) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(47, 111, 176, ${p.alpha * 0.75})`;
      ctx.fill();
    }

    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.hypot(dx, dy);

        if (dist < 120) {
          const opacity = 1 - dist / 120;
          ctx.strokeStyle = `rgba(47, 111, 176, ${opacity * 0.12})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  };

  const handleResize = () => {
    resize();
    spawn();
  };

  handleResize();
  window.addEventListener("resize", handleResize);
  requestAnimationFrame(draw);
}

const seedEl = document.getElementById("dataset-seed");
const seedList = seedEl ? JSON.parse(seedEl.textContent || "[]") : [];
const datasetInfo = {};

const buildPlaceholderInfo = (name) => ({
  caption: `${name} · 结果图示意`,
});

if (seedList.length) {
  seedList.forEach((name) => {
    datasetInfo[name] = buildPlaceholderInfo(name);
  });
} else {
  datasetInfo["示例数据集"] = {
    caption: "示例数据集 · 结果图示意",
  };
}

let tabs = document.querySelectorAll("#dataset-tabs .tab");
const figureCaption = document.getElementById("figure-caption");

const updateDataset = async (key) => {
  const data = datasetInfo[key];
  if (!data) return;
  
  // 跟踪当前数据集
  currentDataset = key;

  if (figureCaption) figureCaption.textContent = data.caption;

  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.dataset === key);
  });

  try {
    const response = await fetch(`/api/dataset/${encodeURIComponent(key)}`);
    const result = await response.json();
    
    if (result.plot_points && result.plot_points.length > 0) {
      const xs = result.plot_points.map((p) => p[0]);
      const ys = result.plot_points.map((p) => p[1]);
      const zs = result.plot_points.map((p) => p[2]);

      plotTrace = {
        x: xs,
        y: ys,
        z: zs,
        mode: "markers",
        type: "scatter3d",
        marker: {
          size: 2.5,
          color: "#2f6fb0",
          opacity: 0.8,
        },
      };

      renderPlot();
    }
  } catch (error) {
    console.error("Failed to load dataset:", error);
  }
};

const bindTabEvents = () => {
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      updateDataset(tab.dataset.dataset);
    });
  });
};

bindTabEvents();

const fileInput = document.getElementById("dataset-input");
if (fileInput) {
  fileInput.addEventListener("change", (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    files.forEach((file) => {
      const name = file.name;
      if (!datasetInfo[name]) {
        datasetInfo[name] = {
          caption: `${name} · 结果图示意`,
        };

        const button = document.createElement("button");
        button.className = "tab";
        button.type = "button";
        button.dataset.dataset = name;
        button.textContent = name;
        const container = document.getElementById("dataset-tabs");
        if (container) container.appendChild(button);
      }
    });

    tabs = document.querySelectorAll("#dataset-tabs .tab");
    bindTabEvents();
    updateDataset(files[0].name);
    fileInput.value = "";
  });
}

const plotDataEl = document.getElementById("plot-data");
const plotContainer = document.getElementById("plot3d");
const scheduleWrap = document.getElementById("schedule-wrap");
const scheduleAxesPlot = document.getElementById("schedule-plot-axes");
const scheduleDataPlot = document.getElementById("schedule-plot-data");
const particleLayer = document.getElementById("particle-layer");
const scheduleDataEl = document.getElementById("schedule-data");
let showingSchedule = false;
let plotTrace = null;
let plotLayout = null;
let currentDataset = null;
let currentScheduleData = null;
const plotConfig = { responsive: true, displaylogo: false };
const scheduleConfig = { responsive: true, displaylogo: false };

const loadScheduleData = async (datasetName, pointIndex) => {
  try {
    const response = await fetch(`/api/schedule/${encodeURIComponent(datasetName)}/${pointIndex}`);
    const result = await response.json();
    if (result.processes) {
      currentScheduleData = result.processes;
      return result.processes;
    }
  } catch (error) {
    console.error("Failed to load schedule data:", error);
  }
  return null;
};

const bindPlotClick = () => {
  if (!plotContainer || !plotContainer.on) return;
  if (plotContainer.removeAllListeners) {
    plotContainer.removeAllListeners("plotly_click");
  }
  plotContainer.on("plotly_click", async (data) => {
    if (data.points && data.points.length > 0) {
      const pointIndex = data.points[0].pointNumber;
      const datasetName = currentDataset || document.querySelector("#dataset-tabs .tab.active")?.dataset.dataset;
      if (datasetName) {
        const scheduleData = await loadScheduleData(datasetName, pointIndex);
        if (scheduleData) {
          showSchedule(scheduleData);
        }
      }
    }
  });
};

const renderPlot = () => {
  if (!window.Plotly || !plotContainer || !plotTrace || !plotLayout) return;
  window.Plotly.purge(plotContainer);
  window.Plotly.newPlot(plotContainer, [plotTrace], plotLayout, plotConfig);
  window.Plotly.Plots.resize(plotContainer);
  bindPlotClick();
};

const spawnParticles = () => {
  if (!particleLayer) return;
  particleLayer.innerHTML = "";
  const count = 18;
  for (let i = 0; i < count; i += 1) {
    const particle = document.createElement("span");
    particle.className = "particle";
    particle.style.top = `${10 + Math.random() * 80}%`;
    particle.style.left = `${10 + Math.random() * 80}%`;
    particle.style.animationDelay = `${Math.random() * 0.3}s`;
    particle.style.transform = `translateX(${Math.random() * 10}px)`;
    particleLayer.appendChild(particle);
  }
};

const playScheduleReveal = () => {
  if (!scheduleWrap) return;
  scheduleWrap.classList.remove("reveal");
  void scheduleWrap.offsetWidth;
  scheduleWrap.classList.add("reveal");
  spawnParticles();
};

const showSchedule = (scheduleData = null) => {
  if (!plotContainer || !scheduleWrap || !scheduleDataPlot || !scheduleAxesPlot) return;
  
  // 如果已经在显示调度图，直接更新数据而不阻止
  const wasShowingSchedule = showingSchedule;
  showingSchedule = true;
  
  scheduleWrap.style.display = "block";
  plotContainer.style.display = "none";
  
  requestAnimationFrame(() => {
    renderSchedulePlot(scheduleData);
    // 只有在从plot切换过来时才播放动画
    if (!wasShowingSchedule) {
      playScheduleReveal();
    }
  });
};

const showPlot = () => {
  if (!plotContainer || !scheduleWrap) return;
  if (!showingSchedule) return;
  showingSchedule = false;
  scheduleWrap.style.display = "none";
  plotContainer.style.display = "block";
  requestAnimationFrame(() => {
    renderPlot();
  });
};
if (plotDataEl && plotContainer && window.Plotly) {
  const rawPoints = JSON.parse(plotDataEl.textContent || "[]");
  if (rawPoints.length) {
    const xs = rawPoints.map((p) => p[0]);
    const ys = rawPoints.map((p) => p[1]);
    const zs = rawPoints.map((p) => p[2]);

    plotTrace = {
      x: xs,
      y: ys,
      z: zs,
      mode: "markers",
      type: "scatter3d",
      marker: {
        size: 2.5,
        color: "#2f6fb0",
        opacity: 0.85,
      },
    };

    plotLayout = {
      autosize: true,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: {
          title: "X",
          color: "rgba(20,20,20,0.95)",
          linecolor: "rgba(20,20,20,0.95)",
          gridcolor: "rgba(20,20,20,0.2)",
          zerolinecolor: "rgba(20,20,20,0.34)",
        },
        yaxis: {
          title: "Y",
          color: "rgba(20,20,20,0.95)",
          linecolor: "rgba(20,20,20,0.95)",
          gridcolor: "rgba(20,20,20,0.2)",
          zerolinecolor: "rgba(20,20,20,0.34)",
        },
        zaxis: {
          title: "Z",
          color: "rgba(20,20,20,0.95)",
          linecolor: "rgba(20,20,20,0.95)",
          gridcolor: "rgba(20,20,20,0.2)",
          zerolinecolor: "rgba(20,20,20,0.34)",
        },
        bgcolor: "rgba(0,0,0,0)",
      },
    };

    renderPlot();
    const placeholder = plotContainer.parentElement?.querySelector(".figure-placeholder");
    if (placeholder) placeholder.style.display = "none";

    bindPlotClick();
  }
}

if (scheduleWrap) {
  scheduleWrap.addEventListener("click", () => {
    showPlot();
  });
}

// 缩放功能
const figureCanvas = document.getElementById("figure-canvas");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const zoomResetBtn = document.getElementById("zoom-reset");
const zoomLevelDisplay = document.getElementById("zoom-level");

let currentZoom = 1.0;
const minZoom = 0.5;
const maxZoom = 3.0;
const zoomStep = 0.2;

const updateZoom = () => {
  if (!figureCanvas) return;
  figureCanvas.style.transform = `scale(${currentZoom})`;
  if (zoomLevelDisplay) {
    zoomLevelDisplay.textContent = `${Math.round(currentZoom * 100)}%`;
  }
  // 调整容器高度以适应缩放后的内容
  const parent = figureCanvas.parentElement;
  if (parent) {
    const baseHeight = parent.clientHeight;
    parent.style.minHeight = `${baseHeight * currentZoom}px`;
  }
};

if (zoomInBtn) {
  zoomInBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentZoom < maxZoom) {
      currentZoom = Math.min(maxZoom, currentZoom + zoomStep);
      updateZoom();
    }
  });
}

if (zoomOutBtn) {
  zoomOutBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentZoom > minZoom) {
      currentZoom = Math.max(minZoom, currentZoom - zoomStep);
      updateZoom();
    }
  });
}

if (zoomResetBtn) {
  zoomResetBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentZoom = 1.0;
    updateZoom();
  });
}

// 鼠标滚轮缩放
if (figureCanvas) {
  figureCanvas.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
      const newZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));
      if (newZoom !== currentZoom) {
        currentZoom = newZoom;
        updateZoom();
      }
    }
  }, { passive: false });
}

const buildScheduleTraces = (records) => {
  // 学术论文风格的100色配色方案 - 更淡、不透明的颜色
  // 使用HSL色彩空间生成，降低饱和度、提高亮度，使颜色更柔和
  const generateAcademicPalette = (count) => {
    const colors = [];
    const goldenRatio = 0.618033988749895;
    let hue = 0.5; // 起始色相
    
    for (let i = 0; i < count; i++) {
      // 使用黄金角增量确保色相均匀分布
      hue = (hue + goldenRatio) % 1;
      
      // 饱和度: 35-50% (降低饱和度，颜色更淡)
      const saturation = 0.5 + (i % 4) * 0.04;
      
      // 亮度: 65-78% (提高亮度，颜色更柔和)
      const lightness = 0.65 + (i % 5) * 0.03;
      
      // 转换为HSL字符串，添加不透明度0.85
      const h = hue * 360;
      const s = saturation * 100;
      const l = lightness * 100;
      
      colors.push(`hsla(${h.toFixed(1)}, ${s.toFixed(1)}%, ${l.toFixed(1)}%, 100)`);
    }
    return colors;
  };
  
  const jobPalette = generateAcademicPalette(100);

  const rows = records
    .map((item) => {
      const [factory, process, machine, job, start, end] = item;
      return {
        factory,
        process,
        machine,
        job,
        start,
        end,
        y: `F${factory} S${process} M${machine}`,
      };
    })
    .sort((a, b) => {
      if (a.factory !== b.factory) return a.factory - b.factory;
      if (a.process !== b.process) return a.process - b.process;
      if (a.machine !== b.machine) return a.machine - b.machine;
      return a.start - b.start;
    });

  const uniqueJobs = [...new Set(rows.map((row) => row.job))].sort((a, b) => a - b);
  const jobColorMap = new Map(uniqueJobs.map((job, index) => [job, jobPalette[index % jobPalette.length]]));
  const makeBarTrace = (subset) => ({
    type: "bar",
    orientation: "h",
    y: subset.map((row) => row.y),
    x: subset.map((row) => row.end - row.start),
    base: subset.map((row) => row.start),
    marker: {
      color: subset.map((row) => jobColorMap.get(row.job) || "#2f6fb0"),
      line: { color: "rgba(60, 80, 100, 0.25)", width: 0.6 },
      opacity: 1,
    },
    hovertemplate:
      "工件 %{customdata[0]}<br>工厂 %{customdata[1]} · 工序 %{customdata[2]} · 机器 %{customdata[3]}<br>开始 %{customdata[4]} 结束 %{customdata[5]}<br>时长 %{customdata[6]}<extra></extra>",
    customdata: subset.map((row) => [
      row.job,
      row.factory,
      row.process,
      row.machine,
      row.start,
      row.end,
      (row.end - row.start).toFixed(1),
    ]),
  });
  const makeTextTrace = (subset, fontSize) => ({
    type: "scatter",
    mode: "text",
    x: subset.map((row) => row.start + (row.end - row.start) / 2),
    y: subset.map((row) => row.y),
    text: subset.map((row) => `J${row.job}`),
    textposition: "middle center",
    textfont: { color: "rgba(20, 20, 20, 0.95)", size: fontSize, family: "Noto Serif SC, serif" },
    cliponaxis: false,
    hoverinfo: "skip",
    showlegend: false,
  });

  const longRows = rows.filter((row) => row.end - row.start >= 4);
  const midRows = rows.filter((row) => row.end - row.start >= 2 && row.end - row.start < 4);
  const shortRows = rows.filter((row) => row.end - row.start < 2);
  return [
    makeBarTrace(rows),
    makeTextTrace(longRows, 11),
    makeTextTrace(midRows, 9),
    makeTextTrace(shortRows, 7),
  ].filter((trace) => trace.y.length > 0);
};

const pickTimeTick = (maxEnd, recordCount = 60) => {
  // 根据数据规模动态调整时间间隔
  // recordCount: 记录数，用于判断数据规模
  // 小规模数据 (20工件): 约60条记录
  // 中规模数据 (40工件): 约200条记录  
  // 大规模数据 (100工件): 约500条记录
  
  // 基础时间间隔根据maxEnd计算
  let baseTick;
  if (maxEnd <= 20) baseTick = 5;
  else if (maxEnd <= 50) baseTick = 10;
  else if (maxEnd <= 100) baseTick = 20;
  else if (maxEnd <= 200) baseTick = 25;
  else if (maxEnd <= 400) baseTick = 50;
  else if (maxEnd <= 800) baseTick = 100;
  else baseTick = 200;
  
  // 根据数据规模调整时间间隔
  // 数据量越大，时间间隔越大，避免刻度拥挤
  let scaleFactor = 1;
  if (recordCount <= 60) {
    scaleFactor = 1; // 小规模数据，保持基础间隔
  } else if (recordCount <= 200) {
    scaleFactor = 1; // 中规模数据，保持基础间隔
  } else if (recordCount <= 500) {
    scaleFactor = 1; // 大规模数据，保持基础间隔
  } else {
    scaleFactor = 2; // 超大规模数据，间隔翻倍
  }
  
  // 计算最终时间间隔，确保在图表上显示8-12个刻度
  const targetTicks = 8;
  const calculatedTick = maxEnd / targetTicks;
  
  // 将计算值规整到标准刻度 (1, 2, 5, 10, 20, 25, 50, 100, 200, 500)
  const standardTicks = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000];
  let finalTick = standardTicks[0];
  
  for (const tick of standardTicks) {
    if (calculatedTick <= tick * 1.5) {
      finalTick = tick;
      break;
    }
  }
  
  return finalTick;
};

const buildScheduleLayout = (records) => {
  let maxEnd = 0;
  const factoryRowMap = new Map();
  records.forEach((item) => {
    const [factory, process, machine, , , end] = item;
    if (!factoryRowMap.has(factory)) factoryRowMap.set(factory, new Map());
    const rows = factoryRowMap.get(factory);
    const key = `${process}-${machine}`;
    rows.set(key, { process, machine });
    if (end > maxEnd) maxEnd = end;
  });

  const sortedFactories = Array.from(factoryRowMap.keys()).sort((a, b) => a - b);
  const factoryBlocks = sortedFactories.map((factory) => {
    const rows = Array.from(factoryRowMap.get(factory).values())
      .sort((a, b) => {
        if (a.process !== b.process) return a.process - b.process;
        return a.machine - b.machine;
      })
      .map((row) => ({
        full: `F${factory} S${row.process} M${row.machine}`,
        short: `S${row.process} M${row.machine}`,
      }));
    return { factory, rows };
  });

  const yCategoriesTopDown = [];
  const tickVals = [];
  const tickText = [];

  factoryBlocks.forEach((block, index) => {
    block.rows.forEach((row) => {
      yCategoriesTopDown.push(row.full);
      tickVals.push(row.full);
      tickText.push(row.short);
    });
    if (index !== factoryBlocks.length - 1) {
      const gap = `F${block.factory}-GAP`;
      yCategoriesTopDown.push(gap);
      tickVals.push(gap);
      tickText.push("");
    }
  });

  const yCategories = [...yCategoriesTopDown].reverse();
  const totalCategories = yCategoriesTopDown.length;
  const visibleRows = yCategoriesTopDown.filter((row) => !row.includes("-GAP")).length;
  const yTickSize =
    visibleRows <= 12 ? 12 :
    visibleRows <= 18 ? 11 :
    visibleRows <= 24 ? 10 :
    visibleRows <= 32 ? 9 :
    visibleRows <= 44 ? 8 : 7;
  const factoryAnnotations = [];
  const factoryShapes = [];
  let cursor = 0;
  factoryBlocks.forEach((block, index) => {
    const startIndex = cursor;
    const centerIndex = startIndex + (block.rows.length - 1) / 2;
    const yTop = 1 - startIndex / totalCategories;
    const yBottom = 1 - (startIndex + block.rows.length) / totalCategories;
    const y = 1 - (centerIndex + 0.5) / totalCategories;
    factoryShapes.push({
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "paper",
      y0: yBottom,
      y1: yTop,
      line: { width: 0 },
      fillcolor: "rgba(255, 255, 255, 0)",
      layer: "below",
    });
    factoryAnnotations.push({
      text: `Factory ${block.factory}`,
      x: -0.18,
      xref: "paper",
      yref: "paper",
      y,
      showarrow: false,
      align: "right",
      font: { color: "rgba(45, 63, 84, 0.8)", size: 12, family: "Noto Serif SC, serif" },
    });
    cursor += block.rows.length;
    if (index !== factoryBlocks.length - 1) cursor += 1;
  });

  const dtick = pickTimeTick(maxEnd, records.length);

  return {
    autosize: true,
    margin: { l: 200, r: 20, t: 56, b: 58 },
    paper_bgcolor: "rgba(249, 252, 255, 0.92)",
    plot_bgcolor: "rgba(249, 252, 255, 0.92)",
    hovermode: "closest",
    hoverlabel: {
      bgcolor: "rgba(255, 255, 255, 0.98)",
      bordercolor: "rgba(47,111,176,0.24)",
      font: { color: "rgba(31,42,55,0.94)", size: 12 },
    },
    barmode: "overlay",
    bargap: 0.28,
    bargroupgap: 0.1,
    title: {
      text: "Factory Schedule (Right-shift strategy)",
      font: { color: "rgba(31,42,55,0.9)", size: 16, family: "Noto Serif SC, serif" },
      x: 0.5,
    },
    xaxis: {
      title: {
        text: "Time (hours)",
        standoff: 12,
        font: { color: "rgba(20,20,20,0.95)", size: 15, family: "Noto Serif SC, serif" },
      },
      tick0: 0,
      dtick,
      range: [0, Math.ceil(maxEnd + dtick * 0.5)],
      tickformat: ".0f",
      tickangle: 0,
      gridcolor: "rgba(20,20,20,0.12)",
      zerolinecolor: "rgba(20,20,20,0.4)",
      zerolinewidth: 1.5,
      tickfont: { color: "rgba(20,20,20,0.9)", size: 12 },
      showgrid: true,
      gridwidth: 1,
      ticks: "outside",
      ticklen: 6,
      tickwidth: 1.5,
      tickcolor: "rgba(20,20,20,0.5)",
    },
    yaxis: {
      title: "",
      categoryorder: "array",
      categoryarray: yCategories,
      tickmode: "array",
      tickvals: tickVals,
      ticktext: tickText,
      gridcolor: "rgba(20,20,20,0.14)",
      tickfont: { color: "rgba(20,20,20,0.92)", size: yTickSize },
      automargin: true,
    },
    shapes: factoryShapes,
    annotations: [
      ...factoryAnnotations,
    ],
    showlegend: false,
  };
};

const renderSchedulePlot = (scheduleData = null) => {
  if (!scheduleDataPlot || !scheduleAxesPlot || !window.Plotly) return;
  
  // 使用传入的schedule数据，如果没有则使用页面中嵌入的数据
  let raw = scheduleData;
  if (!raw || !raw.length) {
    if (!scheduleDataEl) return;
    raw = JSON.parse(scheduleDataEl.textContent || "[]");
  }
  if (!raw || !raw.length) return;
  const scheduleTraces = buildScheduleTraces(raw);
  const scheduleLayout = buildScheduleLayout(raw);
  const yCats = scheduleLayout?.yaxis?.categoryarray || [];
  const axesTrace = {
    type: "scatter",
    x: yCats.map(() => 0),
    y: yCats,
    mode: "markers",
    marker: { opacity: 0 },
    hoverinfo: "skip",
    showlegend: false,
  };
  const axesLayout = {
    ...scheduleLayout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };
  const dataLayout = {
    ...scheduleLayout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    title: { text: "" },
    xaxis: { ...scheduleLayout.xaxis, showgrid: false, showticklabels: false, zeroline: false, title: "" },
    yaxis: { ...scheduleLayout.yaxis, showgrid: false, showticklabels: false, title: "" },
  };

  window.Plotly.react(scheduleAxesPlot, [axesTrace], axesLayout, scheduleConfig);
  window.Plotly.react(scheduleDataPlot, scheduleTraces, dataLayout, scheduleConfig);
  window.Plotly.Plots.resize(scheduleAxesPlot);
  window.Plotly.Plots.resize(scheduleDataPlot);
};
