/*
 * clinical_patterns.js
 *
 * Maps ALREADY-MEASURED kinematic findings to established PT/kinesiology
 * "probable contributing factor" patterns (Janda's crossed syndromes,
 * Trendelenburg-test logic, the dynamic-valgus/hip-abductor link, etc).
 *
 * IMPORTANT — WHAT THIS IS AND ISN'T:
 * These are well-documented movement-pattern correlations from clinical
 * literature, applied to real measured angles/ratios. They are NOT muscle
 * tests, NOT EMG, NOT a measurement of muscle activation or strength —
 * video cannot see that. Every output here is phrased as "consistent
 * with" a probable factor, paired with a manual test to confirm it. This
 * is an interpretive layer on top of measurements, not a new measurement,
 * and should always be rendered as clearly visually distinct from the
 * measured-findings sections.
 */

export function getProbableContributingFactors(findings) {
    const patterns = [];
    const add = (title, likelyFactors, recommendedCheck) => {
        patterns.push({ title, likelyFactors, recommendedCheck });
    };

    // --- Upper crossed syndrome pattern (Janda) ---
    if (findings.forwardHeadAngle != null && findings.forwardHeadAngle < 20) {
        // (smaller angle = more forward head carriage, per this app's convention)
        add(
            "Forward Head Posture Pattern",
            "Consistent with an upper-crossed-syndrome pattern: probable tightness of upper trapezius / levator scapulae / pectorals, with probable weakness of deep neck flexors and lower trapezius / serratus anterior.",
            "Confirm with deep neck flexor endurance test and manual length testing of pectoralis minor / upper trapezius."
        );
    }

    // --- Dynamic knee valgus (squat or gait stance) ---
    const valgusRatio = findings.kneeValgusRatio;
    if (valgusRatio != null && valgusRatio < 0.85) {
        add(
            "Dynamic Knee Valgus Pattern",
            "One of the most consistently reported correlates of dynamic knee valgus in the sports-medicine literature is hip abductor / external rotator weakness (gluteus medius and deep external rotators), sometimes alongside relative vastus medialis obliquus (VMO) underactivity or an overactive tensor fasciae latae / IT band.",
            "Confirm with single-leg squat test, hip abductor manual muscle test, and Trendelenburg test."
        );
    }
    if (findings.frontGaitKneeValgus && findings.frontGaitKneeValgus.side) {
        const s = findings.frontGaitKneeValgus.side;
        add(
            `Dynamic Knee Valgus During Gait Stance (${s})`,
            `Same pattern as above, observed specifically during the loaded stance phase on the ${s.toLowerCase()} leg — classically associated with ${s.toLowerCase()}-side hip abductor/external rotator insufficiency.`,
            "Confirm with single-leg squat test and manual hip abductor strength testing on this side."
        );
    }

    // --- Trendelenburg / hip drop pattern ---
    if (findings.trendelenburgPattern && findings.trendelenburgPattern.side) {
        const s = findings.trendelenburgPattern.side;
        add(
            `Contralateral Pelvic Drop During ${s} Single-Limb Stance`,
            `This is the classic Trendelenburg pattern — a pelvic drop on the swing side during ${s.toLowerCase()}-leg stance is the textbook sign of ${s.toLowerCase()}-side hip abductor (gluteus medius) insufficiency.`,
            "Confirm with a formal Trendelenburg test and manual gluteus medius strength testing."
        );
    }

    // --- Knee hyperextension ---
    if (findings.kneeHyperextensionFlag) {
        add(
            "Knee Hyperextension Pattern",
            "Habitual knee hyperextension (\"locking\") during standing or stance is often a compensation for quadriceps weakness (passively stabilizing the joint instead of active control), and can also reflect posterior capsule / ligamentous laxity.",
            "Confirm with quadriceps manual muscle test and passive knee extension end-feel assessment."
        );
    }

    // --- Excessive forward trunk lean ---
    if (findings.excessiveTrunkLeanFlag) {
        add(
            "Excessive Forward Trunk Lean Pattern",
            "A larger-than-expected forward trunk lean during a squat or gait stance is a well-documented compensation for either restricted ankle dorsiflexion (shifting the center of mass forward to stay over the base of support) or quadriceps weakness (reducing knee extensor demand by loading the hip extensors instead).",
            "Confirm with weight-bearing ankle dorsiflexion lunge test and quadriceps manual muscle test."
        );
    }

    // --- Heel lift / limited ankle dorsiflexion ---
    if (findings.heelLiftDetected || findings.limitedAnkleDorsiflexionFlag) {
        add(
            "Limited Ankle Dorsiflexion Pattern",
            "Heels rising during a squat, or reduced forward knee travel over the foot, is classically associated with gastrocnemius/soleus tightness restricting ankle dorsiflexion range.",
            "Confirm with weight-bearing lunge (knee-to-wall) test for ankle dorsiflexion ROM."
        );
    }

    // --- Rear-view shank/heel alignment (pronation-associated pattern) ---
    if (findings.rearShankAlignment && findings.rearShankAlignment.side) {
        const s = findings.rearShankAlignment.side;
        const dir = findings.rearShankAlignment.direction || "medial";
        add(
            `${dir === "medial" ? "Medial (Pronation-Associated)" : "Lateral (Supination-Associated)"} Shank Lean During ${s} Stance`,
            dir === "medial"
                ? `A medial lean of the lower leg during stance, viewed from behind, is the classic visual proxy for a dynamic pronation pattern. This is often discussed as part of a kinetic chain involving both foot/ankle control (tibialis posterior) and proximal hip control, not foot structure alone.`
                : `A lateral lean of the lower leg during stance, viewed from behind, is associated with a supination-biased loading pattern.`,
            "Confirm with a weight-bearing foot posture index assessment and hip abductor strength testing on this side."
        );
    }

    // --- General asymmetry flags (kept deliberately non-specific — too many
    // possible causes to name a muscle with confidence) ---
    if (findings.limbAsymmetryFlag) {
        add(
            "Left/Right Asymmetry Pattern",
            "A meaningful difference between limbs was observed. This can reflect strength or mobility asymmetry, a leg-length difference, prior injury, or simply habitual loading preference — too many possible causes to narrow down from movement alone.",
            "Recommend bilateral strength and mobility comparison, and a leg-length assessment if not already done."
        );
    }

    return patterns;
}

// Rendering helper shared by both report pages — visually distinct from
// measured-findings rows (purple accent, no red/amber/green flag scale,
// since these are interpretive hypotheses, not pass/fail measurements).
export function renderContributingFactorsHtml(patterns) {
    if (!patterns.length) return "";
    let html = `<div class="report-section-header" style="color:#b388ff;">Probable Contributing Factors — Clinical Pattern Correlations</div>`;
    html += `<div class="report-row" style="border-left-color:#b388ff; background: rgba(179,136,255,0.08);">
        <div class="report-note" style="font-style:italic;">These are established movement-pattern correlations from PT/kinesiology literature (e.g. Janda's crossed syndromes, Trendelenburg-test logic), applied to the measurements above — not a muscle test, not EMG, not a diagnosis. Each should be confirmed with the manual test noted.</div>
    </div>`;
    for (const p of patterns) {
        html += `
        <div class="report-row" style="border-left-color:#b388ff;">
            <div class="report-row-top"><span class="report-label">${p.title}</span></div>
            <div class="report-note">${p.likelyFactors}</div>
            <div class="report-note" style="margin-top:4px;"><b>Suggested confirmation:</b> ${p.recommendedCheck}</div>
        </div>`;
    }
    return html;
}
