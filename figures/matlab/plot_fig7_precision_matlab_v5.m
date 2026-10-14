%% Fig. 7 / Frozen-checkpoint precision screening -- dual-scale layout v5
% MATLAB R2021b compatible. No toolboxes required.
%
% Compared with v4, each of the three panels now has TWO horizontally
% separated views of the SAME archived data:
%   left:  complete range, all seven interventions and seed/SD marks;
%   right: near-zero magnification for FP weights, W16A8 and nonlinear ops.
% The detail column is NOT an independent sample or reanalysis.
% No new data, statistics, or figure values are calculated.
% Panel-specific scales remain explicit; all displayed marks/SD are checked.
% Figure is publication-oriented: no overlapping inset obscures results.
%
clearvars; close all; clc;
sourceDir = fileparts(mfilename('fullpath'));
if isempty(sourceDir), sourceDir = pwd; end
sourceFile = fullfile(sourceDir, 'fig_precision_evidence_source.json');
assert(exist(sourceFile, 'file') == 2, 'Cannot find JSON: %s', sourceFile);
S = jsondecode(fileread(sourceFile));
assert(isfield(S, 'all_grid_summaries_22_variants'), ...
    'Unexpected Fig. 7 JSON structure.');

% The paper intentionally shows only these seven archived probes.
keys = {'restore_all_activations','w8a16','restore_all_weights', ...
        'w16a8','restore_eq_back','restore_mixing', ...
        'restore_nonlinear_operators'};
labels = {'FP activations','W8A16 probe','FP weights', ...
          'W16A8 probe','Back block','Front + back','Nonlinear ops.'};
tags = {'b1_longcp','b2_longcp','b2_shortcp'};
headers = {'(a) BPSK / CP16','(b) QPSK / CP16','(c) QPSK / CP4'};
rows = S.all_grid_summaries_22_variants;
assert(numel(rows) == 66, 'Expected 66 archived records (22 per config).');

% Read the recorded per-seed deltas and verify the archived means / sample SD.
mu = nan(7,3);
sd = nan(7,3);
individual = nan(7,3,3);  % intervention, seed, configuration
for c = 1:3
    for j = 1:7
        matching = find(strcmp({rows.tag}, tags{c}) & strcmp({rows.variant}, keys{j}));
        assert(numel(matching) == 1, 'Missing/duplicate %s / %s', tags{c}, keys{j});
        r = rows(matching);
        v = double(r.raw_seed_grid_mean_deltas(:));
        m = double(r.mean_delta_ber);
        s = double(r.sample_sd_delta_ber);
        assert(numel(v)==3 && all(isfinite(v)) && isfinite(m) && isfinite(s), ...
            'Invalid data in %s / %s', tags{c}, keys{j});
        assert(abs(mean(v)-m) < 1e-11 && abs(std(v,0)-s) < 1e-11, ...
            'Mean or sample SD differs from source: %s / %s', tags{c}, keys{j});
        mu(j,c) = 1000*m;
        sd(j,c) = 1000*s;
        individual(j,:,c) = (1000*v)';
    end
end

% Publication-safe palette.
ink       = [0.16 0.25 0.36];
seedInk   = [0.33 0.42 0.52];
copper    = [0.79 0.34 0.17];
lineGray  = [0.85 0.88 0.90];
zeroGray  = [0.56 0.61 0.66];
rowShade  = [0.972 0.979 0.985];
dividerC  = [0.70 0.74 0.78];

figW = 18.3; 
figH = 19.7;
fig = figure('Color','w','Units','centimeters', ...
    'Position',[2 2 figW figH], ...
    'PaperUnits','centimeters','PaperSize',[figW figH], ...
    'PaperPosition',[0 0 figW figH], ...
    'Renderer','painters','InvertHardcopy','off', ...
    'Name','Fig. 7 / Precision screening (v5: main + near-zero detail)');

left = 4.65;
width = 8.50;        % Full-range view
zoomGap = 0.50;      % Physical separation between full/zoom axes (cm)
zoomLeft = left + width + zoomGap;
zoomWidth = 3.70;   % Near-zero detail view; overall right edge stays 17.35 cm
height = 3.82;
bottoms = [15.05 9.85 4.65];   % larger inter-panel gaps than v2
% Panel-specific horizontal axis presets.  Same BER unit in all panels.
% Adjust these three rows if you want to change the displayed ranges.
xRanges = [-1.2 0.2;    % panel (a): BPSK / CP16
           -2.6 0.2;    % panel (b): QPSK / CP16
           -4.4 0.2];   % panel (c): QPSK / CP4
xTickSteps = [0.2 0.5 1.0];
xTickFormats = {'%.1f','%.1f','%.0f'};

seedDy = [-0.22 -0.075 0.075];
meanDy = 0.235;
seedShape = {'o','s','^'};

for c = 1:3
    xRange = xRanges(c,:);
    tickStep = xTickSteps(c);

    % The ranges are approximate presets, not permission to crop observations.
    % Include all three seed points plus both ends of every mean +/- sample SD.
    seedValues = individual(:,:,c);
    actualMin = min([seedValues(:); mu(:,c)-sd(:,c)]);
    actualMax = max([seedValues(:); mu(:,c)+sd(:,c)]);
    pad = max(0.025*(xRange(2)-xRange(1)), 0.025);
    presetRange = xRange;
    if actualMin-pad < xRange(1)
        xRange(1) = floor((actualMin-pad)/tickStep)*tickStep;
    end
    if actualMax+pad > xRange(2)
        xRange(2) = ceil((actualMax+pad)/tickStep)*tickStep;
    end
    if any(abs(xRange-presetRange) > 1e-10)
        fprintf('Panel %s: x-range auto-extended [%.3f, %.3f] -> [%.3f, %.3f] to avoid clipping.\n', ...
            headers{c}, presetRange(1), presetRange(2), xRange(1), xRange(2));
    end
    assert(actualMin > xRange(1) && actualMax < xRange(2), ...
        'Panel %s has data/SD outside axis range', headers{c});
    firstTick = ceil((xRange(1)-1e-10)/tickStep);
    lastTick = floor((xRange(2)+1e-10)/tickStep);
    xTicks = (firstTick:lastTick)*tickStep;
    xTicks(abs(xTicks) < 1e-10) = 0;
    tickLabels = arrayfun(@(v) sprintf(xTickFormats{c},v), xTicks, ...
        'UniformOutput',false);
    tickLabels(abs(xTicks) < 1e-10) = {'0'};

    ax = axes('Parent',fig,'Units','centimeters', ...
        'Position',[left bottoms(c) width height]);
    hold(ax,'on');
    set(ax,'FontName','Arial','FontSize',9.1, ...
        'LineWidth',0.70,'TickDir','out','TickLength',[0.009 0.009], ...
        'Box','off','Layer','top','YDir','reverse', ...
        'XColor',ink,'YColor',ink,'Color','w');
    xlim(ax,xRange);
    ylim(ax,[0.50 7.53]);
    set(ax,'XTick',xTicks,'XTickLabel',tickLabels, ...
        'YTick',1:7,'YTickLabel',labels);
    set(ax,'TickLabelInterpreter','none');
    t = title(ax,headers{c},'FontName','Arial','FontSize',10.4, ...
        'FontWeight','bold','Color',ink,'Interpreter','none');
    t.Units = 'normalized';
    t.Position(2) = 1.02;

    for j = 1:7
        if mod(j,2)==0
            patch(ax,[xRange(1) xRange(2) xRange(2) xRange(1)], ...
                [j-.48 j-.48 j+.48 j+.48], rowShade, ...
                'EdgeColor','none','HandleVisibility','off');
        end
    end

    for xt = xTicks
        if xt ~= 0
            line(ax,[xt xt],[0.50 7.53],'Color',lineGray, ...
                'LineWidth',0.38,'LineStyle',':','HandleVisibility','off');
        end
    end
    line(ax,[0 0],[0.50 7.53],'Color',zeroGray, ...
        'LineWidth',0.95,'LineStyle','--','HandleVisibility','off');
    line(ax,xRange,[4.5 4.5],'Color',dividerC, ...
        'LineWidth',0.85,'HandleVisibility','off');

    for j = 1:7
        for k = 1:3
            plot(ax,individual(j,k,c), j+seedDy(k), seedShape{k}, ...
                'LineStyle','none','Color',seedInk, ...
                'MarkerFaceColor','w','MarkerSize',5.6, ...
                'LineWidth',0.9,'HandleVisibility','off');
        end
        x0 = mu(j,c); spread = sd(j,c); y = j + meanDy;
        line(ax,[x0-spread x0+spread],[y y], ...
            'Color',copper,'LineWidth',1.35,'HandleVisibility','off');
        capH = 0.072;
        line(ax,[x0-spread x0-spread],[y-capH y+capH], ...
            'Color',copper,'LineWidth',1.0,'HandleVisibility','off');
        line(ax,[x0+spread x0+spread],[y-capH y+capH], ...
            'Color',copper,'LineWidth',1.0,'HandleVisibility','off');
        plot(ax,x0,y,'d','LineStyle','none', ...
            'Color',copper,'MarkerFaceColor',copper, ...
            'MarkerSize',6.0,'LineWidth',0.9,'HandleVisibility','off');
    end

    % ---------------------------------------------------------------
    % Near-zero enlarged detail: original seed values and sample SDs,
    % plotted AGAIN (not duplicated data/independent observations).
    % Only weight restoration, W16A8, and nonlinear-control rows are
    % repeated.  The full-range plot on the left remains complete.
    % ---------------------------------------------------------------
    focusRows = [3 4 7];
    zoomPresets = [-0.16 0.12;  % (a) BPSK/CP16
                   -0.48 0.12;  % (b) QPSK/CP16
                   -0.30 0.12]; % (c) QPSK/CP4
    zoomSteps = [0.05 0.20 0.10];
    zRange = zoomPresets(c,:);
    zStep = zoomSteps(c);
    seedPart = individual(focusRows,:,c);
    zMin = min([seedPart(:); mu(focusRows,c)-sd(focusRows,c)]);
    zMax = max([seedPart(:); mu(focusRows,c)+sd(focusRows,c)]);
    zPad = max(0.025, 0.03*(zRange(2)-zRange(1)));
    beforeZoom = zRange;
    if zMin-zPad < zRange(1)
        zRange(1) = floor((zMin-zPad)/zStep)*zStep;
    end
    if zMax+zPad > zRange(2)
        zRange(2) = ceil((zMax+zPad)/zStep)*zStep;
    end
    if any(abs(zRange-beforeZoom) > 1e-10)
        fprintf('Panel %s: detail x-range widened [%.3f, %.3f] -> [%.3f, %.3f].\n', ...
            headers{c}, beforeZoom(1),beforeZoom(2),zRange(1),zRange(2));
    end
    assert(zMin > zRange(1) && zMax < zRange(2), ...
        'Detail view would clip seed data or mean +/- SD in %s',headers{c});

    zTicks = (ceil((zRange(1)-1e-10)/zStep):floor((zRange(2)+1e-10)/zStep))*zStep;
    zTicks(abs(zTicks)<1e-10) = 0;
    zTickLabels = arrayfun(@(x) sprintf('%.2g',x),zTicks,'UniformOutput',false);
    zTickLabels(abs(zTicks)<1e-10) = {'0'};

    zax = axes('Parent',fig,'Units','centimeters', ...
        'Position',[zoomLeft bottoms(c) zoomWidth height]);
    hold(zax,'on');
    set(zax,'FontName','Arial','FontSize',8.2, ...
        'LineWidth',0.65,'TickDir','out','TickLength',[0.015 0.015], ...
        'Box','on','Layer','top','YDir','reverse', ...
        'XColor',ink,'YColor',dividerC,'Color','w');
    xlim(zax,zRange); ylim(zax,[0.50 7.53]);
    set(zax,'XTick',zTicks,'XTickLabel',zTickLabels,'YTick',[], ...
        'TickLabelInterpreter','none');

    for j = 1:7
        if mod(j,2)==0
            patch(zax,[zRange(1) zRange(2) zRange(2) zRange(1)], ...
                [j-.48 j-.48 j+.48 j+.48],rowShade, ...
                'EdgeColor','none','HandleVisibility','off');
        end
    end
    for xt = zTicks
        if xt ~= 0
            line(zax,[xt xt],[0.50 7.53],'Color',lineGray, ...
                'LineWidth',0.38,'LineStyle',':','HandleVisibility','off');
        end
    end
    if zRange(1) < 0 && zRange(2) > 0
        line(zax,[0 0],[0.50 7.53],'Color',zeroGray, ...
            'LineWidth',0.95,'LineStyle','--','HandleVisibility','off');
    end
    line(zax,zRange,[4.5 4.5],'Color',dividerC, ...
        'LineWidth',0.85,'HandleVisibility','off');

    for rowIndex = 1:numel(focusRows)
        j = focusRows(rowIndex);
        for k = 1:3
            plot(zax,individual(j,k,c),j+seedDy(k),seedShape{k}, ...
                'LineStyle','none','Color',seedInk, ...
                'MarkerFaceColor','w','MarkerSize',5.6, ...
                'LineWidth',0.9,'HandleVisibility','off');
        end
        x0 = mu(j,c); spread = sd(j,c); y = j+meanDy;
        line(zax,[x0-spread x0+spread],[y y], ...
            'Color',copper,'LineWidth',1.35,'HandleVisibility','off');
        capH = 0.072;
        line(zax,[x0-spread x0-spread],[y-capH y+capH], ...
            'Color',copper,'LineWidth',1.0,'HandleVisibility','off');
        line(zax,[x0+spread x0+spread],[y-capH y+capH], ...
            'Color',copper,'LineWidth',1.0,'HandleVisibility','off');
        plot(zax,x0,y,'d','LineStyle','none', ...
            'Color',copper,'MarkerFaceColor',copper, ...
            'MarkerSize',6.0,'LineWidth',0.9,'HandleVisibility','off');
    end
    text(zax,0.5,1.04,'Near-zero detail', ...
        'Units','normalized','FontName','Arial','FontSize',8.2, ...
        'FontWeight','normal','Color',ink, ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'Interpreter','none');
end

% Common measurement-unit label; each panel has its own x-axis range.
annotation(fig,'textbox',[0.267 0.112 0.69 0.028], ...
    'String','Candidate - W8A8 matrix: BER change (x 10^{-3})', ...
    'FontName','Arial','FontSize',9.2, ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'Color',ink,'EdgeColor','none','Interpreter','tex', ...
    'FitBoxToText','off');

% Clarify that the right-hand views magnify only three existing rows.
annotation(fig,'textbox',[0.255 0.150 0.70 0.022], ...
    'String','Right column: enlarged view of FP weights, W16A8 and nonlinear operators (same data)', ...
    'FontName','Arial','FontSize',7.6, ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'Color',ink,'EdgeColor','none','Interpreter','none', ...
    'FitBoxToText','off');

% Dedicated legend strip.
legendAx = axes('Parent',fig,'Units','centimeters', ...
    'Position',[4.70 0.74 12.55 0.70], ...
    'Color','none','Visible','off');
hold(legendAx,'on'); xlim(legendAx,[0 12.55]); ylim(legendAx,[0 1]);
legendX = [0.10 2.30 4.50 6.95];
legendName = {'Seed 1','Seed 2','Seed 3','Mean +/- sample SD'};
for k = 1:4
    x = legendX(k);
    if k <= 3
        plot(legendAx,x,0.5,seedShape{k},'LineStyle','none', ...
            'Color',seedInk,'MarkerFaceColor','w', ...
            'MarkerSize',6.0,'LineWidth',0.95);
    else
        line(legendAx,[x-.22 x+.22],[.5 .5], ...
            'Color',copper,'LineWidth',1.35);
        plot(legendAx,x,.5,'d','LineStyle','none', ...
            'Color',copper,'MarkerFaceColor',copper,'MarkerSize',6.2);
    end
    text(legendAx,x+.28,.5,legendName{k}, ...
        'FontName','Arial','FontSize',8.7, ...
        'Color',ink,'VerticalAlignment','middle','Interpreter','none');
end

outputBase = fullfile(sourceDir, 'fig7_matlab_publication_v5');
try
    savefig(fig,[outputBase '.fig']);
    print(fig,[outputBase '.pdf'],'-dpdf','-painters');
    print(fig,[outputBase '.svg'],'-dsvg','-painters');
    print(fig,[outputBase '.png'],'-dpng','-r600');
catch ME
    fprintf(2,'Figure export failed: %s\n',ME.message);
    rethrow(ME);
end
fprintf('Fig. 7 v5 exported: %s.[fig/pdf/svg/png]\n',outputBase);
fprintf('Verified 21 contrasts, 63 individual seed values, with recorded means and sample SDs.\n');
fprintf('Layout changes only: full-range left and near-zero detail right in each panel; no clipping in either view.\n');
