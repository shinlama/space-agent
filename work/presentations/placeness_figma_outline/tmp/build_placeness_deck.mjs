import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = "C:/Users/jiwon/Desktop/연구/space-agent";
const outDir = path.join(root, "outputs");
const previewDir = path.join(root, "work/presentations/placeness_figma_outline/tmp/preview");
const finalPptx = path.join(outDir, "placeness_quantification_20_slide_outline.pptx");

await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const deck = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const C = {
  bg: "#FBFDFF",
  ink: "#172033",
  muted: "#556173",
  line: "#D8E3EF",
  blue: "#2F6EA6",
  sky: "#E8F3FF",
  skyStrong: "#7BB8F0",
  green: "#9AD35B",
  greenSoft: "#EFF8E4",
  mango: "#F5BE4B",
  mangoSoft: "#FFF4DC",
  coral: "#F19C79",
  white: "#FFFFFF",
  pale: "#F6FAFE",
};

const noLine = { style: "solid", fill: "none", width: 0 };
const thinLine = { style: "solid", fill: C.line, width: 1 };

function addBox(slide, left, top, width, height, fill = C.white, line = thinLine, radius = "rounded-xl", name = "box") {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left, top, width, height },
    fill,
    line,
    borderRadius: radius,
  });
}

function addRect(slide, left, top, width, height, fill, name = "rect") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left, top, width, height },
    fill,
    line: noLine,
  });
}

function addText(slide, text, left, top, width, height, fontSize, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name || "text",
    position: { left, top, width, height },
    fill: "none",
    line: noLine,
  });
  shape.text = text;
  shape.text.style = {
    fontSize,
    bold: Boolean(opts.bold),
    color: opts.color || C.ink,
    alignment: opts.align || "left",
  };
  return shape;
}

function addBullets(slide, items, left, top, width, height, fontSize = 23, color = C.ink) {
  return addText(slide, items.map((item) => `• ${item}`).join("\n"), left, top, width, height, fontSize, { color });
}

function addHeader(slide, index, title, section) {
  slide.background.fill = C.bg;
  addRect(slide, 0, 0, 1280, 14, C.blue, "top-accent");
  addText(slide, section.toUpperCase(), 72, 46, 360, 26, 14, { bold: true, color: C.blue });
  addText(slide, String(index).padStart(2, "0"), 1120, 38, 88, 34, 21, { bold: true, color: "#A8B6C6", align: "right" });
  addText(slide, title, 72, 88, 880, 58, 36, { bold: true, color: C.ink });
}

function card(slide, title, body, left, top, width, height, accent = C.blue) {
  addBox(slide, left, top, width, height, C.white, thinLine, "rounded-xl", "card");
  addRect(slide, left, top, 7, height, accent, "card-accent");
  addText(slide, title, left + 24, top + 24, width - 44, 34, 20, { bold: true });
  addText(slide, body, left + 24, top + 66, width - 44, height - 84, 17, { color: C.muted });
}

function chip(slide, label, left, top, width, fill, color = C.ink) {
  addBox(slide, left, top, width, 42, fill, noLine, "rounded-2xl", "chip");
  addText(slide, label, left + 12, top + 10, width - 24, 18, 14, { bold: true, color, align: "center" });
}

function arrow(slide, left, top, width = 56, color = "#8EA2B8") {
  addText(slide, "→", left, top, width, 44, 32, { bold: true, color, align: "center" });
}

function newSlide(index, title, section) {
  const slide = deck.slides.add();
  slide.name = `${String(index).padStart(2, "0")}. ${title}`;
  addHeader(slide, index, title, section);
  return slide;
}

function sectionTag(slide, text, fill, left, top, width = 180) {
  chip(slide, text, left, top, width, fill);
}

// 1
{
  const slide = deck.slides.add();
  slide.name = "01. Title";
  slide.background.fill = C.bg;
  addRect(slide, 0, 0, 26, 720, C.blue, "left-accent");
  addText(slide, "An Approach to Quantifying\nPlaceness Using Spatial\nReview Text Data", 84, 110, 810, 220, 54, { bold: true });
  addText(slide, "Using User Reviews of Cafés in Seoul", 86, 344, 650, 38, 23, { color: C.blue });
  addText(slide, "Ji-Won Shin, Jin-Kook Lee", 86, 520, 520, 30, 19, { bold: true });
  addText(slide, "Department of Interior Architecture, Yonsei University", 86, 554, 720, 28, 16, { color: C.muted });
  chip(slide, "Placeness", 86, 622, 140, C.sky);
  chip(slide, "Review text data", 246, 622, 190, C.greenSoft);
  chip(slide, "Quantification", 456, 622, 170, C.mangoSoft);
}

// 2
{
  const slide = newSlide(2, "Research Focus", "Background");
  addBox(slide, 118, 242, 1044, 230, C.sky, { style: "solid", fill: "#9DC6EA", width: 2 }, "rounded-2xl", "research-focus-statement");
  addText(slide, "This study aims to quantify placeness,\na qualitative and subjective concept.", 178, 303, 924, 112, 38, { bold: true, color: C.blue, align: "center" });
}

// 3
{
  const slide = newSlide(3, "Research Highlights", "Background");
  addBullets(slide, [
    "Placeness has mostly been evaluated through subjective methods such as surveys and observation, which motivates a data-based and LLM-assisted quantitative approach.",
    "Placeness is structured into three dimensions and ten detailed factors based on placeness theory and prior studies.",
    "A logical scoring framework is proposed to extract, classify, and quantify placeness factors from spatial review data.",
    "The empirical analysis uses Google Maps reviews collected from cafés in Seoul.",
    "The quantified results can be extended to place recommendation, search, and comparison.",
  ], 86, 170, 1050, 360, 21);
  addBox(slide, 180, 578, 920, 66, C.mangoSoft, { style: "solid", fill: "#F4D38C", width: 1 }, "rounded-xl");
  addText(slide, "Core contribution: converting qualitative place experience into comparable placeness indicators", 214, 600, 852, 24, 20, { bold: true, color: "#9A6500", align: "center" });
}

// 4
{
  const slide = newSlide(4, "Placeness Theory", "Literature Review");
  addText(slide, "Placeness is understood as a multidimensional concept formed through the interaction of physical environment, human activity, and meaning.", 86, 166, 1010, 70, 28, { bold: true });
  card(slide, "Physical characteristics", "Observable spatial and environmental qualities", 94, 332, 314, 190, C.blue);
  card(slide, "Activity-related characteristics", "Behaviors, uses, and social interactions within space", 482, 332, 314, 190, C.green);
  card(slide, "Meaning-related characteristics", "Memory, preference, symbolism, and identity formed through experience", 870, 332, 314, 190, C.mango);
}

// 5
{
  const slide = newSlide(5, "Components of Placeness", "Literature Review");
  addBox(slide, 96, 170, 1088, 400, C.pale, { style: "solid", fill: "#9DC6EA", width: 2 }, "rounded-2xl", "figure-placeholder");
  addText(slide, "Figure placeholder", 512, 246, 260, 30, 22, { bold: true, color: C.blue, align: "center" });
  card(slide, "Physical", "Spatial form\nEnvironment\nAccessibility", 170, 338, 245, 136, C.blue);
  card(slide, "Activity", "Staying\nWorking\nMeeting\nInteracting", 518, 338, 245, 136, C.green);
  card(slide, "Meaning", "Memory\nPreference\nSymbolism\nIdentity", 866, 338, 245, 136, C.mango);
  addText(slide, "Placeness emerges through the interaction of these three components.", 170, 610, 940, 34, 23, { bold: true, align: "center" });
}

// 6
{
  const slide = newSlide(6, "Relph’s Core Components of Placeness", "Literature Review");
  addText(slide, "Relph explains the identity of place through the relationship among physical setting, activities, and meanings.", 90, 166, 1030, 56, 26, { bold: true });
  card(slide, "Physical setting", "Material and spatial conditions that form the setting of experience.", 114, 310, 310, 190, C.blue);
  card(slide, "Activities", "Actions and patterns of use that occur in the place.", 486, 310, 310, 190, C.green);
  card(slide, "Meanings", "Values, memories, and interpretations attached to the place.", 858, 310, 310, 190, C.mango);
  addText(slide, "Reference: Edward Relph, Place and Placelessness (1976)", 114, 604, 860, 28, 18, { color: C.muted });
}

// 7
{
  const slide = newSlide(7, "Placeness Factors in Prior Studies", "Literature Review");
  addText(slide, "Placeness factors were selected and similar items were integrated according to the environmental characteristics of commercial spaces.", 74, 150, 1080, 38, 20, { bold: true, color: C.ink });
  const x = 54;
  const y = 208;
  const widths = [42, 160, 350, 140, 468];
  const rowH = 50;
  const headerH = 34;
  const headers = ["No.", "Study", "Title", "Target", "Placeness factors"];
  let cx = x;
  headers.forEach((h, i) => {
    addRect(slide, cx, y, widths[i], headerH, "#E9EEF5");
    addBox(slide, cx, y, widths[i], headerH, "none", { style: "solid", fill: "#7C8796", width: 0.6 }, "rounded-none");
    addText(slide, h, cx + 5, y + 8, widths[i] - 10, 14, 11, { bold: true, color: C.ink, align: "center" });
    cx += widths[i];
  });
  const studies = [
    ["1", "Guo, Hwang & Lee\n(2025)", "User perception analysis of museum spatial design based on place characteristics", "Museum", "[Physical] accessibility, territoriality, attractiveness, comfort\n[Activity] interactivity, convenience\n[Meaning] symbolism, educational value"],
    ["2", "Na, Lee & Lee\n(2025)", "Analysis of placeness of major places in Seoul using text mining and large language models", "71 major places\nin Seoul", "Symbolism, culturality, everydayness, specificity, vitality, quietness, naturalness"],
    ["3", "Sun & Yu\n(2021)", "Dimension and formation of placeness of commercial public space in city center", "Commercial\nfacility", "[Physical] traffic convenience, visibility, safety, comfort, openness\n[Activity] activity, seating, environmental pleasure\n[Meaning] regional landmarkness, functional context"],
    ["4", "Lyu\n(1996)", "A study on embodiment factors of sense of place in interior space", "Residential\nspace", "Centrality, boundary, control, safety, comfort, symbolism, orientation, sociality, interior design elements"],
    ["5", "Lee & Kim\n(2025)", "Spatial design strategy of large cafés from the perspective of complex cultural space", "Eight large\ncafés", "Complexity, accessibility, participation, symbolism, expertise, historicity, experience, diversity, aesthetics, culture, openness, continuity"],
    ["6", "Waxman\n(2008)", "The Coffee Shop: Social and physical factors influencing place attachment", "Three cafés\nin Alabama", "[Physical] hygiene, scent, lighting, furniture, exterior view, seating layout\n[Activity] staying, territoriality, stability, anonymity, productivity, interaction\n[Meaning] emotional bond"],
  ];
  studies.forEach((row, r) => {
    cx = x;
    const top = y + headerH + r * rowH;
    row.forEach((cell, i) => {
      addRect(slide, cx, top, widths[i], rowH, r % 2 === 0 ? C.white : "#F7FAFC");
      addBox(slide, cx, top, widths[i], rowH, "none", { style: "solid", fill: "#C9D3DF", width: 0.5 }, "rounded-none");
      addText(slide, cell, cx + 5, top + 7, widths[i] - 10, rowH - 10, i === 4 ? 8.3 : 8.8, { color: C.ink, align: i === 0 || i === 3 ? "center" : "left" });
      cx += widths[i];
    });
  });
  addBox(slide, 310, 600, 660, 62, C.sky, { style: "solid", fill: "#9DC6EA", width: 1 }, "rounded-lg");
  addText(slide, "Integrated into 3 dimensions and 10 placeness factors for commercial spaces", 338, 620, 604, 20, 18, { bold: true, color: C.blue, align: "center" });
}

// 8
{
  const slide = newSlide(8, "Reconstructed Placeness Factors for Commercial Spaces", "Literature Review");
  addText(slide, "Detailed placeness factors were reconstructed according to the spatial environmental characteristics of commercial facilities.", 74, 150, 1080, 38, 20, { bold: true, color: C.ink });
  addBox(slide, 330, 220, 240, 34, C.white, { style: "solid", fill: "#7C8796", width: 1 }, "rounded-none");
  addText(slide, "Formative elements of placeness", 350, 230, 200, 12, 11, { bold: true, align: "center" });
  addBox(slide, 620, 220, 330, 34, C.white, { style: "solid", fill: "#7C8796", width: 1 }, "rounded-none");
  addText(slide, "Spatial environmental characteristics of commercial spaces", 640, 230, 290, 12, 11, { bold: true, align: "center" });
  const groups = [
    ["Physical characteristics", ["Aesthetics", "Openness", "Sensory experience", "Comfort", "Accessibility"], 292, C.blue],
    ["Activity-related characteristics", ["Activity", "Interactivity"], 492, C.green],
    ["Meaning-related characteristics", ["Symbolism", "Memory and preference", "Local identity"], 586, C.mango],
  ];
  groups.forEach(([label, factors, top, accent]) => {
    const h = factors.length * 34;
    addBox(slide, 330, top + h / 2 - 21, 240, 42, "#F2F4F7", { style: "solid", fill: "#7C8796", width: 1 }, "rounded-none");
    addText(slide, label, 352, top + h / 2 - 6, 196, 14, 13, { bold: true, align: "center", color: C.ink });
    addText(slide, "─", 574, top + h / 2 - 10, 44, 18, 18, { color: "#7C8796", align: "center" });
    factors.forEach((factor, i) => {
      const fy = top + i * 34;
      addBox(slide, 620, fy, 330, 34, "#F2F4F7", { style: "solid", fill: "#7C8796", width: 1 }, "rounded-none");
      addRect(slide, 620, fy, 6, 34, accent);
      addText(slide, factor, 652, fy + 9, 250, 12, 12, { color: C.ink, align: "center" });
    });
  });
}

// 9
{
  const slide = newSlide(9, "Detailed Placeness Factor Framework", "Literature Review");
  card(slide, "Physical characteristics", "Aesthetics\nOpenness\nSensory experience\nAccessibility\nComfort", 82, 210, 320, 330, C.blue);
  card(slide, "Activity-related characteristics", "Activity\nInteraction", 480, 210, 320, 330, C.green);
  card(slide, "Meaning-related characteristics", "Symbolism\nMemory and preference\nLocal identity", 878, 210, 320, 330, C.mango);
  addText(slide, "The ten factors serve as the evaluation framework for placeness quantification.", 116, 594, 1040, 42, 24, { bold: true, align: "center" });
}

// 10
{
  const slide = newSlide(10, "Definitions and Keywords of Placeness Factors", "Literature Review");
  const rows = [
    ["Aesthetics", "visual quality, design, atmosphere"],
    ["Openness", "large windows, views, spaciousness"],
    ["Sensory experience", "light, smell, sound, taste, tactile comfort"],
    ["Accessibility", "location, transit, parking, entry"],
    ["Comfort", "cleanliness, seating, temperature, crowding"],
    ["Activity", "work, reading, staying, gathering"],
    ["Interaction", "staff, companions, social exchange"],
    ["Symbolism", "landmark, uniqueness, representative image"],
    ["Memory & preference", "revisit intention, attachment, favorite place"],
    ["Local identity", "neighborhood character, local context"],
  ];
  rows.forEach(([factor, desc], i) => {
    const col = i < 5 ? 0 : 1;
    const x = col === 0 ? 86 : 662;
    const y = 166 + (i % 5) * 86;
    addBox(slide, x, y, 520, 62, C.white, thinLine, "rounded-lg");
    addText(slide, factor, x + 18, y + 14, 176, 24, 17, { bold: true, color: col === 0 ? C.blue : C.green });
    addText(slide, desc, x + 202, y + 14, 286, 28, 15, { color: C.muted });
  });
}

// 11
{
  const slide = newSlide(11, "Review Expressions by Placeness Factor", "Literature Review");
  addText(slide, "Each factor was defined so that it can be identified through expressions that appear in user reviews.", 86, 162, 1040, 48, 25, { bold: true });
  const examples = [
    ["Openness", "“The space feels open because the windows are large.”"],
    ["Accessibility", "“Parking was inconvenient.”"],
    ["Comfort", "“The restroom was clean and well managed.”"],
    ["Activity", "“It is good for studying or working.”"],
    ["Memory & preference", "“I want to visit this café again.”"],
  ];
  examples.forEach(([factor, quote], i) => {
    const y = 260 + i * 70;
    addBox(slide, 130, y, 1020, 48, i % 2 === 0 ? C.sky : C.mangoSoft, noLine, "rounded-lg");
    addText(slide, factor, 158, y + 13, 220, 20, 16, { bold: true, color: C.blue });
    addText(slide, quote, 400, y + 12, 710, 22, 16, { color: C.ink });
  });
}

// 12
{
  const slide = newSlide(12, "Research Target and Scope", "Research Scope");
  addText(slide, "The empirical target is cafés in Seoul, selected as a case for testing the review-based placeness quantification approach.", 86, 166, 1050, 60, 27, { bold: true });
  card(slide, "Study area", "Seoul, South Korea", 114, 310, 310, 190, C.blue);
  card(slide, "Spatial target", "Cafés and bakery/café-related commercial spaces", 486, 310, 310, 190, C.green);
  card(slide, "Scope limitation", "The result should be interpreted as an empirical case, not a generalization to all space types.", 858, 310, 310, 190, C.mango);
}

// 13
{
  const slide = newSlide(13, "Overview of Research", "Method");
  card(slide, "Input", "Text review data", 112, 314, 250, 158, C.blue);
  arrow(slide, 380, 362);
  card(slide, "Placeness Quantification System", "Placeness factor extraction and quantification", 470, 274, 340, 238, C.green);
  arrow(slide, 828, 362);
  card(slide, "Output", "Placeness scores", 918, 314, 250, 158, C.mango);
  addBox(slide, 170, 560, 940, 70, C.sky, { style: "solid", fill: "#9DC6EA", width: 1 }, "rounded-xl");
  addText(slide, "Figure placeholder: overview of placeness-based text review quantification approach", 210, 584, 860, 26, 21, { bold: true, color: C.blue, align: "center" });
}

// 14
{
  const slide = newSlide(14, "Internal Modules of the Quantification System", "Method");
  card(slide, "Placeness factor mapping module", "Identifies review phrases that serve as evidence and maps them to the ten placeness factors.", 114, 238, 360, 190, C.blue);
  arrow(slide, 496, 302);
  card(slide, "Placeness factor scoring module", "Classifies the evaluation direction of mapped evidence and calculates factor-level scores.", 610, 238, 360, 190, C.green);
  addBox(slide, 248, 514, 784, 92, C.sky, { style: "solid", fill: "#9DC6EA", width: 1 }, "rounded-xl");
  addText(slide, "Prior-study-based placeness factor evaluation framework", 302, 544, 676, 28, 23, { bold: true, color: C.blue, align: "center" });
}

// 15
{
  const slide = newSlide(15, "Internal Process of the Quantification System", "Method");
  card(slide, "1. Review preprocessing", "Clean review format and filter reviews unrelated to spatial experience.", 72, 282, 260, 160, C.blue);
  arrow(slide, 344, 330, 42);
  card(slide, "2. Phrase segmentation", "Separate reviews into sentence, phrase, or clause-level units.", 386, 282, 260, 160, C.green);
  arrow(slide, 658, 330, 42);
  card(slide, "3. Factor mapping", "Map each evidence phrase to placeness factors using the evaluation framework.", 700, 282, 260, 160, C.mango);
  arrow(slide, 972, 330, 42);
  card(slide, "4. Scoring", "Classify direction and calculate factor score and mention share.", 1010, 282, 210, 160, C.coral);
  addText(slide, "Preprocessing and segmentation are included inside the mapping module.", 178, 552, 924, 34, 23, { bold: true, color: C.muted, align: "center" });
}

// 16
{
  const slide = newSlide(16, "Placeness Factor Mapping Module", "Method");
  addText(slide, "The GPT-4o-mini API is used to map review phrases to placeness factors through an instruction prompt.", 86, 166, 1040, 58, 27, { bold: true });
  const items = [
    ["Task", "Map review phrases to one or more placeness factors."],
    ["Definitions", "Provide definitions of the ten placeness factors."],
    ["Keywords", "Provide keywords and review-observable expressions."],
    ["Examples", "Provide positive and negative examples for each factor."],
    ["Output", "Return mapped factor and supporting phrase."],
  ];
  items.forEach(([title, body], i) => {
    const x = 96 + (i % 3) * 382;
    const y = i < 3 ? 296 : 486;
    card(slide, title, body, x, y, 310, 132, [C.blue, C.green, C.mango, C.coral, C.skyStrong][i]);
  });
}

// 17
{
  const slide = newSlide(17, "Placeness Factor Scoring Module", "Method");
  card(slide, "Phrase-level score", "Positive phrase: +1\nNeutral or mixed phrase: 0\nNegative phrase: -1", 88, 210, 340, 210, C.blue);
  card(slide, "Factor score", "(positive evidence count - negative evidence count) / total evidence count for the factor", 470, 210, 340, 210, C.green);
  card(slide, "Mention share", "Evidence count for each factor / all placeness evidence for the place", 852, 210, 340, 210, C.mango);
  addBox(slide, 176, 510, 928, 86, C.sky, { style: "solid", fill: "#9DC6EA", width: 1 }, "rounded-xl");
  addText(slide, "Score range: -1 to +1. Negative values are preserved when negative evidence is dominant.", 220, 538, 840, 30, 22, { bold: true, color: C.blue, align: "center" });
}

// 18
{
  const slide = newSlide(18, "Data Used in This Study", "Data");
  card(slide, "1. Public commercial data", "Café and bakery/donut businesses in Seoul were filtered from public commercial district data.", 76, 226, 262, 180, C.blue);
  arrow(slide, 350, 286, 40);
  card(slide, "2. Sample places", "2,500 cafés were sampled by selecting 100 places from each of Seoul’s 25 districts.", 390, 226, 262, 180, C.green);
  arrow(slide, 664, 286, 40);
  card(slide, "3. Google Maps reviews", "34,671 reviews were collected from places with available reviews.", 704, 226, 262, 180, C.mango);
  arrow(slide, 978, 286, 40);
  card(slide, "4. Placeness mappings", "17,641 reviews were mapped to placeness factors across 1,486 places.", 1014, 226, 190, 180, C.coral);
  addText(slide, "Data may be updated as review collection expands.", 116, 560, 1040, 32, 20, { color: C.muted, align: "center" });
}

// 19
{
  const slide = newSlide(19, "Demo and Application", "Data");
  addText(slide, "Demo", 86, 164, 220, 38, 28, { bold: true, color: C.blue });
  addText(slide, "https://space-agent.streamlit.app/", 86, 210, 640, 28, 20, { color: C.muted });
  addBullets(slide, [
    "Shows the process from the placeness factor framework to review mapping results and score calculation.",
    "Allows users to examine which review phrases support each factor score.",
    "Can be extended to place comparison, spatial search, and personalized recommendation.",
  ], 86, 288, 520, 180, 21);
  addBox(slide, 680, 162, 500, 360, C.pale, { style: "solid", fill: "#9DC6EA", width: 2 }, "rounded-xl", "demo-screenshot-placeholder");
  addText(slide, "Replace with updated\ndemo screenshot", 760, 294, 340, 82, 26, { bold: true, color: C.blue, align: "center" });
}

// 20
{
  const slide = newSlide(20, "Conclusion and Limitations", "Conclusion");
  addBullets(slide, [
    "This study confirmed that placeness can be quantified using user-generated text review data.",
    "The proposed scoring framework produces comparable indicators based on placeness factors.",
    "Review-based quantification can complement fieldwork- and survey-based evaluation methods.",
    "The system can be extended to place recommendation, spatial search, and place comparison.",
    "Limitations remain because the empirical target is cafés in Seoul and online reviews do not represent every user experience.",
  ], 96, 170, 1040, 310, 22);
  addBox(slide, 170, 560, 940, 76, C.mangoSoft, { style: "solid", fill: "#F4D38C", width: 1 }, "rounded-xl");
  addText(slide, "From qualitative place experience to comparable quantitative placeness indicators", 218, 584, 844, 30, 24, { bold: true, color: "#9A6500", align: "center" });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(previewDir, `${stem}.png`), await deck.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text());
}

await writeBlob(path.join(previewDir, "deck-montage.webp"), await deck.export({ format: "webp", montage: true, scale: 1 }));
const snapshot = await deck.inspect({ kind: "slide,textbox,shape,layout", maxChars: 20000 });
await fs.writeFile(path.join(previewDir, "inspect.ndjson"), snapshot.ndjson);

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(finalPptx);

console.log(JSON.stringify({ finalPptx, slideCount: deck.slides.items.length, previewDir }, null, 2));
