function outputs = render_continued_qat(summaryDir, outputDir, exportPdf, auditPython, alignmentAuditor)
%RENDER_CONTINUED_QAT Native MATLAB R2021b figure from audited real CSVs.
%   OUTPUTS = RENDER_CONTINUED_QAT(SUMMARYDIR, NEWOUTPUTDIR, false) writes SVG,
%   PNG and FIG, not PDF. Set exportPdf=true only after the separately managed
%   PDF-operation authorization. No models, stimuli or final tests are run.
%   Optional auditPython/alignmentAuditor run a backend-neutral geometry QA;
%   all scientific drawing and production exports remain native MATLAB.
%   Style-only inheritance: existing manuscript colors/open seed markers.
%   No historical observations, fitted curves or mock observations are used.

if nargin < 3, exportPdf = false; end
if nargin < 4 || isempty(auditPython)
    auditPython = 'D:/anaconda/envs/pytorch_env/python.exe';
end
if nargin < 5 || isempty(alignmentAuditor)
    alignmentAuditor = fullfile(getenv('USERPROFILE'), '.codex', 'skills', ...
        'nature-figure', 'scripts', 'audit_panel_alignment.py');
end
assert(strcmp(version('-release'), '2021b'), ...
    'Use the user-selected MATLAB R2021b, not an alternative plotting backend.');
assert(islogical(exportPdf) && isscalar(exportPdf), 'exportPdf must be a logical scalar.');
summaryDir = canonical_path(summaryDir);
outputDir = canonical_path(outputDir);
scriptPath = [mfilename('fullpath'), '.m'];
revisionRoot = canonical_path(fileparts(fileparts(scriptPath)));
assert(startsWith(lower(outputDir), [lower(revisionRoot), filesep]), ...
    'Keep new figure exports inside the continued_qat_revision_20261011 namespace.');
assert(~isfolder(outputDir) && ~isfile(outputDir), 'Output directory must be new; existing exports are preserved.');
assert(isfile(auditPython) && isfile(alignmentAuditor), 'Required alignment auditor/runtime is missing.');

summaryPath = fullfile(summaryDir, 'summary.json');
manifestPath = fullfile(summaryDir, 'manifest.json');
summary = jsondecode(fileread(summaryPath));
manifest = jsondecode(fileread(manifestPath));
assert(strcmp(summary.status, 'complete') && strcmp(manifest.status, 'complete'), 'Summary is incomplete.');
assert(strcmp(summary.protocol_sha256, manifest.protocol_sha256), 'Summary protocol binding differs.');
assert(strcmp(manifest.independent_audit_status, 'pass'), 'Independent audit must pass before plotting.');
assert(summary.conditions == 72 && summary.training_runs == 18 && ...
    summary.unique_slots == 144000 && summary.unique_bits == 76800000, 'Unexpected evidence budget.');
assert(strcmp(summary.primary_contrast, 'A1_minus_A2'), 'Primary contrast was changed.');

filenames = {'seed_grid_arms.csv', 'seed_grid_contrasts.csv', ...
    'config_arm_summary.csv', 'config_contrast_summary.csv', 'all_points.csv', ...
    'all_contrasts.csv', 'training_costs.csv', 'training_curves.csv', 'summary.json'};
sourceBindings = struct('filename', {}, 'sha256', {});
for index = 1:numel(filenames)
    name = filenames{index};
    path = fullfile(summaryDir, name);
    expected = manifest.outputs.(matlab.lang.makeValidName(name));
    digest = file_sha256(path);
    assert(strcmp(digest, expected.sha256), 'Source-data hash changed: %s', name);
    sourceBindings(index) = struct('filename', name, 'sha256', digest);
end
seedArms = readtable(fullfile(summaryDir, filenames{1}), 'VariableNamingRule', 'preserve');
seedContrasts = readtable(fullfile(summaryDir, filenames{2}), 'VariableNamingRule', 'preserve');
configArms = readtable(fullfile(summaryDir, filenames{3}), 'VariableNamingRule', 'preserve');
configContrasts = readtable(fullfile(summaryDir, filenames{4}), 'VariableNamingRule', 'preserve');
assert(height(seedArms) == 45 && height(seedContrasts) == 54 && ...
    height(configArms) == 15 && height(configContrasts) == 18, 'Incomplete seed/configuration tables.');
assert(all(seedArms.conditions == 8) && all(seedContrasts.conditions == 8), 'The equal-weight grid changed.');
assert(all(configArms.training_seeds == 3) && all(configContrasts.training_seeds == 3), 'Expected three raw seed observations.');

tags = {'b1_longcp', 'b2_longcp', 'b2_shortcp'};
configLabels = {'BPSK, CP16', 'QPSK, CP16', 'QPSK, CP4'};
arms = {'A2', 'A1', 'A1F'};
colors = [0.79 0.33 0.17; 0.12 0.26 0.39; 0.05 0.48 0.48];
seedMarkers = {'o', 's', '^'};
seedColor = [0.35 0.43 0.52];
seedOffsets = [-0.12 0 0.12];
recovery = zeros(3, 3, 3);
primary = zeros(3, 3);
for config = 1:3
    for arm = 1:3
        rows = seedArms(strcmp(string(seedArms.tag), tags{config}) & ...
            strcmp(string(seedArms.arm), arms{arm}), :);
        rows = sortrows(rows, 'training_seed');
        assert(isequal(rows.training_seed(:)', 1:3), 'Raw seed identities are incomplete.');
        calculated = 100 * (rows.grid_mean_ber - rows.a0_grid_mean_ber) ./ rows.a0_grid_mean_ber;
        assert(all(rows.a0_grid_mean_ber > 0) && all(isfinite(calculated)), 'A0-relative percentages are undefined.');
        assert(max(abs(calculated - rows.relative_change_from_a0_percent)) < 1e-10, 'Relative BER source-data identity failed.');
        recovery(config, arm, :) = reshape(calculated, [1 1 3]);
        aggregate = configArms(strcmp(string(configArms.tag), tags{config}) & ...
            strcmp(string(configArms.arm), arms{arm}), :);
        assert(height(aggregate) == 1 && ...
            abs(mean(calculated) - aggregate.mean_seed_relative_change_from_a0_percent) < 1e-10 && ...
            abs(std(calculated, 0) - aggregate.sample_sd_seed_relative_change_from_a0_percent) < 1e-10, ...
            'Relative BER mean/sample SD differs from the complete summary table.');
    end
    rows = seedContrasts(strcmp(string(seedContrasts.tag), tags{config}) & ...
        strcmp(string(seedContrasts.contrast), 'A1_minus_A2'), :);
    rows = sortrows(rows, 'training_seed');
    assert(isequal(rows.training_seed(:)', 1:3), 'Primary raw seeds are incomplete.');
    assert(max(abs(rows.delta_ber - (rows.candidate_grid_mean_ber - rows.reference_grid_mean_ber))) < 1e-14, ...
        'Primary paired BER identity failed.');
    primary(config, :) = rows.delta_ber(:)' * 1000;
    row = configContrasts(strcmp(string(configContrasts.tag), tags{config}) & ...
        strcmp(string(configContrasts.contrast), 'A1_minus_A2'), :);
    assert(height(row) == 1 && abs(row.mean_delta_ber * 1000 - mean(primary(config, :))) < 1e-11 && ...
        abs(row.sample_sd_delta_ber * 1000 - std(primary(config, :), 0)) < 1e-11, 'Primary mean/sample SD differs.');
end
% Preserve the adverse BPSK seed without compressing the QPSK controls.
% Panels b/c share a scale; panel a explicitly uses its own expanded scale.
recoveryLimitsExpanded = shared_limits(recovery(1, :, :));
recoveryLimitsQpsk = shared_limits(recovery(2:3, :, :));
primaryLimits = shared_limits(reshape(primary, [3 1 3]));

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', ...
    'Position', [2 2 18.3 13.8], 'Renderer', 'painters', ...
    'InvertHardcopy', 'off', 'Name', 'Controlled continued-QAT precision comparison');
cleanupFigure = onCleanup(@() close(fig));
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [18.3 13.8], ...
    'PaperPosition', [0 0 18.3 13.8], 'PaperPositionMode', 'manual');
axesHandles = gobjects(6, 1);
panelLabels = gobjects(6, 1);
lefts = [1.50 7.10 12.70];
for config = 1:3
    ax = create_axes(fig, [lefts(config) 8.65 4.65 3.0]);
    axesHandles(config) = ax;
    hold(ax, 'on');
    plot(ax, [0.50 3.60], [0 0], '--', 'Color', [0.60 0.60 0.60], 'LineWidth', 0.65);
    for seed = 1:3
        values = reshape(recovery(config, :, seed), [1 3]);
        plot(ax, (1:3) + seedOffsets(seed), values, '-', 'Color', [0.78 0.81 0.84], 'LineWidth', 0.55);
    end
    for arm = 1:3
        values = reshape(recovery(config, arm, :), [1 3]);
        for seed = 1:3
            plot(ax, arm + seedOffsets(seed), values(seed), seedMarkers{seed}, ...
                'Color', colors(arm, :), 'MarkerFaceColor', 'w', 'MarkerSize', 4.1, 'LineWidth', 0.8);
        end
        errorbar(ax, arm + 0.25, mean(values), std(values, 0), 'd', ...
            'Color', colors(arm, :), 'MarkerFaceColor', colors(arm, :), ...
            'MarkerSize', 4.2, 'LineWidth', 0.95, 'CapSize', 4);
    end
    xlim(ax, [0.50 3.60]);
    if config == 1, ylim(ax, recoveryLimitsExpanded);
    else, ylim(ax, recoveryLimitsQpsk); end
    set(ax, 'XTick', 1:3, 'XTickLabel', arms);
    title(ax, configLabels{config}, 'FontSize', 8.5, 'FontWeight', 'normal');
    if config == 1, ylabel(ax, 'BER change vs A0 (%)', 'FontSize', 8); end
    panelLabels(config) = add_panel_label(ax, char('a' + config - 1));

    ax = create_axes(fig, [lefts(config) 3.55 4.65 3.0]);
    axesHandles(config + 3) = ax;
    hold(ax, 'on');
    plot(ax, [0.5 4.6], [0 0], '--', 'Color', [0.60 0.60 0.60], 'LineWidth', 0.65);
    values = primary(config, :);
    for seed = 1:3
        plot(ax, seed, values(seed), seedMarkers{seed}, 'Color', seedColor, ...
            'MarkerFaceColor', 'w', 'MarkerSize', 4.1, 'LineWidth', 0.8);
    end
    errorbar(ax, 4.10, mean(values), std(values, 0), 'd', 'Color', colors(2, :), ...
        'MarkerFaceColor', colors(2, :), 'MarkerSize', 4.2, 'LineWidth', 0.95, 'CapSize', 4);
    xlim(ax, [0.5 4.6]); ylim(ax, primaryLimits);
    set(ax, 'XTick', [1 2 3 4.1], 'XTickLabel', {'S1', 'S2', 'S3', 'Mean'});
    title(ax, configLabels{config}, 'FontSize', 8.5, 'FontWeight', 'normal');
    if config == 1, ylabel(ax, 'A1 - A2 BER (\times10^{-3})', 'FontSize', 8); end
    panelLabels(config + 3) = add_panel_label(ax, char('d' + config - 1));
end
add_figure_text(fig, [0.50 13.08 17.3 0.42], ...
    'Fixed-budget adaptation versus frozen precision refinement', 9);
add_figure_text(fig, [0.50 12.34 17.3 0.36], ...
    'Controls: BER change from frozen W8A8 (A0); panel a uses an expanded scale', 8);
add_figure_text(fig, [0.50 7.34 17.3 0.36], ...
    'Primary: continued W8A8 (A1) minus original output-only A12 (A2)', 8.5);
add_figure_text(fig, [0.50 2.48 17.3 0.36], ...
    'A2: frozen output-only A12; A1: continued W8A8; A1F: continued W8A8 with frozen ranges.', 8);
add_figure_text(fig, [0.50 1.79 17.3 0.36], ...
    'Open circle / square / triangle: seeds 1 / 2 / 3. Filled diamond and whiskers: mean +/- sample SD.', 8);
add_figure_text(fig, [0.50 1.10 17.3 0.36], ...
    'Each seed averages eight fixed conditions. Negative change means lower BER. SD is not a confidence interval.', 8);
drawnow;
fontHandles = findall(fig, '-property', 'FontSize');
for index = 1:numel(fontHandles)
    assert(get(fontHandles(index), 'FontSize') >= 7.5, 'A rendered font is below the 7.5 pt floor.');
end

mkdir(outputDir);
base = fullfile(outputDir, 'fig_continued_qat_paired_control');
layoutPath = [base, '.panel_layout.json'];
layout = measured_layout(fig, axesHandles, panelLabels);
write_json(layoutPath, layout);
alignmentPath = [base, '.alignment.json'];
overlayPath = [base, '.alignment_overlay.svg'];
command = sprintf('"%s" "%s" "%s" --json-out "%s" --overlay-svg "%s" --tolerance-pt 1.5 --gutter-tolerance-pt 1.5 --require-panel-labels --strict', ...
    quote_safe(auditPython), quote_safe(alignmentAuditor), quote_safe(layoutPath), ...
    quote_safe(alignmentPath), quote_safe(overlayPath));
[auditStatus, auditText] = system(command);
assert(auditStatus == 0, 'Rendered MATLAB panel alignment failed; production exports blocked: %s', auditText);

outputs = struct('fig', [base, '.fig'], 'svg', [base, '.svg'], 'png', [base, '.png'], ...
    'panel_layout', layoutPath, 'alignment', alignmentPath, 'alignment_overlay', overlayPath);
savefig(fig, outputs.fig);
print(fig, outputs.svg, '-dsvg', '-painters');
print(fig, outputs.png, '-dpng', '-r600');
if exportPdf
    outputs.pdf = [base, '.pdf'];
    print(fig, outputs.pdf, '-dpdf', '-painters');
end
% Detect source changes during export rather than silently accepting them.
for index = 1:numel(sourceBindings)
    assert(strcmp(file_sha256(fullfile(summaryDir, sourceBindings(index).filename)), ...
        sourceBindings(index).sha256), 'A source-data file changed during export.');
end
figureSource = struct('status', 'complete', 'backend', 'MATLAB', 'release', version('-release'), ...
    'protocol_sha256', summary.protocol_sha256, 'summary_manifest_sha256', file_sha256(manifestPath), ...
    'renderer_sha256', file_sha256(scriptPath), 'source_bindings', sourceBindings, ...
    'figure_width_cm', 18.3, 'figure_height_cm', 13.8, 'minimum_font_pt', 8, ...
    'top_row_metric', '100*(seed arm eight-condition mean - same-seed A0 mean)/same-seed A0 mean', ...
    'bottom_row_metric', '1000*(seed A1 eight-condition mean - same-seed A2 mean)', ...
    'top_row_values', recovery, 'bottom_row_values', primary, ...
    'top_row_y_limits_a', recoveryLimitsExpanded, 'top_row_y_limits_bc', recoveryLimitsQpsk, ...
    'y_scale_disclosure', 'Panel a expanded; panels b/c share one y-scale; panels d/e/f share one y-scale', ...
    'error_bars', 'sample SD across three raw selected-checkpoint/cohort seeds, not CI', ...
    'primary_contrast', 'A1_minus_A2', 'pdf_export_authorized_by_call', exportPdf, ...
    'geometry_audit', 'backend-neutral audit of measured MATLAB axes; tolerance 1.5 pt; passed', ...
    'visual_qa', 'Pending review of real exported SVG/PNG and, when authorized, rendered PDF', ...
    'scope', 'Software evidence only; no new-checkpoint C/RTL, implementation or board claim');
outputNames = fieldnames(outputs);
figureSource.outputs = struct('filename', {}, 'sha256', {});
for index = 1:numel(outputNames)
    path = outputs.(outputNames{index});
    [~, name, extension] = fileparts(path);
    figureSource.outputs(index) = struct('filename', [name extension], 'sha256', file_sha256(path));
end
outputs.provenance = [base, '.provenance.json'];
write_json(outputs.provenance, figureSource);
disp(outputs);
end


function path = canonical_path(path)
assert(ischar(path) || (isstring(path) && isscalar(path)), 'A scalar path is required.');
path = char(java.io.File(char(path)).getCanonicalPath());
end


function value = quote_safe(value)
value = char(value);
assert(~contains(value, '"') && ~contains(value, newline) && ~contains(value, char(13)), 'Invalid quoted path.');
end


function ax = create_axes(fig, position)
ax = axes('Parent', fig, 'Units', 'centimeters', 'Position', position, ...
    'PositionConstraint', 'innerposition', 'FontName', 'Arial', 'FontSize', 8, ...
    'LineWidth', 0.65, 'TickDir', 'out', 'TickLength', [0.018 0.018], ...
    'Box', 'off', 'Color', 'none', 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15]);
ax.XAxis.Exponent = 0;
ax.YAxis.Exponent = 0;
ax.XGrid = 'off'; ax.YGrid = 'off';
end


function label = add_panel_label(ax, letter)
oldUnits = ax.Units;
ax.Units = 'points'; height = ax.Position(4); ax.Units = oldUnits;
label = text(ax, 0, 0, letter, 'Units', 'points', ...
    'FontName', 'Arial', 'FontSize', 9, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Clipping', 'off');
label.Position = [-12 height + 6 0];
end


function add_figure_text(fig, boxCm, value, fontSize)
box = boxCm ./ [18.3 13.8 18.3 13.8];
annotation(fig, 'textbox', box, 'String', value, 'EdgeColor', 'none', ...
    'Margin', 0, 'FontName', 'Arial', 'FontSize', fontSize, 'Interpreter', 'none', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end


function limits = shared_limits(values)
means = mean(values, 3);
sds = std(values, 0, 3);
extents = [values(:); means(:) - sds(:); means(:) + sds(:); 0];
assert(all(isfinite(extents)), 'Non-finite source data cannot be plotted.');
lower = min(extents); upper = max(extents);
span = upper - lower;
if span == 0, span = 1; end
limits = [lower - 0.12 * span, upper + 0.12 * span];
end


function layout = measured_layout(fig, axesHandles, panelLabels)
% Auto data/plot aspect ratios ensure InnerPosition is the actual plot box.
fig.Units = 'points'; figurePosition = fig.Position;
panels = struct('id', {}, 'bbox_pt', {}, 'grid_id', {}, 'row_start', {}, ...
    'row_stop', {}, 'col_start', {}, 'col_stop', {}, 'panel_label', {}, 'panel_label_anchor_pt', {});
for index = 1:6
    ax = axesHandles(index);
    assert(strcmp(ax.DataAspectRatioMode, 'auto') && strcmp(ax.PlotBoxAspectRatioMode, 'auto'), ...
        'Fixed aspect ratio needs actual plot-box measurement before export.');
    ax.Units = 'points';
    position = ax.InnerPosition;
    panelLabels(index).Units = 'points';
    anchor = position(1:2) + panelLabels(index).Position(1:2);
    panels(index) = struct('id', char('a' + index - 1), ...
        'bbox_pt', [position(1:2), position(1:2) + position(3:4)], ...
        'grid_id', 'continued_qat_2x3', 'row_start', floor((index - 1) / 3), ...
        'row_stop', floor((index - 1) / 3) + 1, 'col_start', mod(index - 1, 3), ...
        'col_stop', mod(index - 1, 3) + 1, 'panel_label', panelLabels(index).String, ...
        'panel_label_anchor_pt', anchor);
end
layout = struct('schema_version', 1, 'backend', 'MATLAB R2021b', ...
    'figure', struct('width_pt', figurePosition(3), 'height_pt', figurePosition(4)), ...
    'panels', panels, 'tolerance_pt', 1.5, 'gutter_tolerance_pt', 1.5, ...
    'measurement', 'drawnow followed by actual axes InnerPosition in point units; both aspect ratios auto', ...
    'row_groups', {{ {'a', 'b', 'c'}, {'d', 'e', 'f'} }}, ...
    'column_groups', {{ {'a', 'd'}, {'b', 'e'}, {'c', 'f'} }});
end


function digest = file_sha256(path)
stream = fopen(path, 'rb');
assert(stream >= 0, 'Cannot read source/export file: %s', path);
cleanupStream = onCleanup(@() fclose(stream));
md = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(stream)
    bytes = fread(stream, 1048576, '*uint8');
    if ~isempty(bytes), md.update(typecast(bytes, 'int8')); end
end
raw = typecast(md.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end


function write_json(path, value)
assert(~isfile(path), 'Refuse to overwrite an existing result: %s', path);
stream = fopen(path, 'w', 'n', 'UTF-8');
assert(stream >= 0, 'Cannot open new JSON result: %s', path);
cleanupStream = onCleanup(@() fclose(stream));
fprintf(stream, '%s\n', jsonencode(value));
end
