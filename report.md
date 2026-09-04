Render review. crates/visual builds clean, but visuals stack has load-bearing bugs. Ordered by damage:
1. Camera-relative double transform — cluster swims, parallax wrong
main.rs:483,498 rebase (world-eye) then render.rs:32 multiplies by full view_proj from camera.rs:55-70 (view already contains -eye). Result: wrong projection (center star stays centered, everything else shifted; bg at r=300 off ~2%, orbit/pan swim).
Fix: either drop rebase (world small, f32 fine) or make view camera-relative: view = look_at_rh(ZERO, tgt-eye, up). Doc docs/09-visual-system.md:62-64 describes rebase but view must match.
2. Double gamma — washed-out, flat contrast
main.rs:246-251 picks sRGB surface. render.rs:73 then applies manual pow(1/2.2). HW sRGB target applies second gamma on store. Net ~1/4.8.
Fix: output linear from tonemap (Reinhard only), let HW convert. Or request non-sRGB surface.
3. Bright-pass UV assumes exact 2x — breaks on odd widths
render.rs:92: fc.xy / dim * 2.0 where dim = full-res HDR, fc = half-res pixel. Only correct if w,h even. gpu.rs:118-119 uses integer /2. Odd width → right/bottom bands sample wrong region (dark stripe / stretched bloom).
Fix: pass texel-size uniform or fullscreen-triangle UV, don't derive UV from source dims.
4. Timestep cap fights default speed — constant judder
main.rs:457-468: speed=4.0, dt=0.004 needs ~17 steps/frame @60fps, cap 4 → acc=0 reset every frame, alpha=0 snap. Motion steppy by design.
Fix: default speed ~1.0, cap 8+, or adaptive cap by N. Current default guarantees hitting spiral-of-death guard.
5. Dead speed field — velocity visuals not wired
sim.rs:195-207 computes speeds()/prev_speeds() per frame, gpu.rs:6-10 carries it, render.rs:9-11,20-35 never reads it. Only misc (temp/mag/size) drives color/size. Wasted alloc + upload, misleading for "speed-based color" work.
Fix: use it or delete it.
6. Physics mismatch Direct vs FMM + no NaN guard
sim.rs:144-146: Direct soft2=1e-6, FmmBackend (sim.rs:38-80) has zero softening (confirmed: no soft in crates/fmm). Same IC diverges; close encounter → huge dv (dt=0.004), inf → GPU garbage quads. Only test is Direct N=128 (sim.rs:258).
Fix: matched softening (or velocity clamp + finite check before upload).
7. Bloom weak + hardcoded + threshold coupled to exposure
render.rs:98-122 single 9-tap separable H+V, radius ~3 half-res texels. Strength *0.5 (render.rs:71) hardcoded, threshold smoothstep(1.0,2.2,lum) (render.rs:94) applied post-exposure (render.rs:49: col*disc*exposure*bright). Slider exposure (main.rs:538) silently moves bloom threshold. Dim stars (bright=0.25+mag*1.75) never cross 1.0.
Fix: threshold pre-exposure (or /exposure), expose strength/radius sliders, add 2-3 mip ping-pong for real glow.
8. Additive-only, no depth — white core, X-ray background
render.rs:153-160 ONE,ONE. Core with 100s overlapping billboards sums to HDR 100+ → Reinhard clips white, bloom explodes. Bg stars (main.rs:595-596, r=300) shine through cluster.
Fix (cheap): clamp per-star HDR, shrink point_size range, or normalize by local density. Real: depth or alpha blend for dense core.
9. Size logic confusing
render.rs:28-29: size = mix(base, tm.z*length(p), step(...)). Sim stars world-size (point_size slider works), bg angular-size (slider ignored) — correct intent, but length(p) only equals eye distance while bug #1 lives. Also tm.z=0 sentinel means any future misc.z=0 star silently switches mode.
Fix: fix #1 first, then explicit flag or separate pipeline vs magic step.
10. Resize holes
main.rs:560-563: Outdated/Lost reconfigures surface but leaves HDR/blur views at old size → stretched composite until next Resized. No ScaleFactorChanged handling.
Fix: route both through one resize(w,h) that rebuilds gpu_state + binds + blur uniforms.
11. Per-frame alloc + full re-upload — hits the wall doc warns about
main.rs:477-518: 4 Vec allocs/frame (prev/curr/speeds), full write_buffer of both buffers (gpu.rs:148-160). @1500 fine (~96KB + 140KB bg not counted); @1e5+ PCIe stall. Contradicts docs/09-visual-system.md:12-27 (state lives on GPU).
Fix: preallocate + reuse buffers, upload only curr (derive prev on GPU via ping-pong), short-term. Compute-shader sim later.
12. Nits
- main.rs:523: bytes_kb excludes bg (~137KB) — overlay under-reports.
- main.rs:23-47: arg parse silent fallback, no --help/validation (--n 0, negative).
- camera.rs:73-90: orbit/pan in physical px — DPI-dependent speed.
- todo.md:5-18 says "fixed-timestep + interpolation" done; bug #4 means interpolation never settles at default speed.
Start: #1, #2, #4. Those three alone explain most "lots of bugs" visuals. Want fixes for #1+#2+#3 first?
