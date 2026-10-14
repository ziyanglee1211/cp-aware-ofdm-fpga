%% Fig. 8 v3 | Output-only A12: isolated precision boundary and confirmation
% MATLAB R2021b compatible; no add-on toolboxes are required.
%
% INPUT : fig_local_a12_evidence_source.json, unchanged, alongside this .m
% OUTPUT: fig8_matlab_publication_v3.[fig/pdf/svg/png]
%
% (a) Operator contract. The back-mixing OUTPUT grid changes to A12; the
%     multiply inputs/weights stay A8/W8 and the four reductions stay INT32.
% (b) Locked, independent confirmation: A12 - FFT W8A8, three independent
%     seeds, all eight EVA/ETU x SNR conditions equally weighted PER SEED;
%     orange diamond and whisker = recorded mean +/- SAMPLE SD (not CI).
% (c) Native-C transfer: only BPSK/CP16 seed-1 subset of the confirmation
%     cohort; recorded slot-bootstrap 95% percentile intervals for eight
%     channel/SNR conditions (NOT a across-seed SD or new independent cohort).
% No new interpolation, data exclusion, threshold calculation or simulation.

clearvars; close all; clc;
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir), rootDir = pwd; end
inputJson = fullfile(rootDir,'fig_local_a12_evidence_source.json');
assert(exist(inputJson,'file') == 2,'Missing JSON file: %s',inputJson);
D = jsondecode(fileread(inputJson));
assert(isfield(D,'confirmation_all_12_grid_rows') && isfield(D,'all_8_c_contrasts'), ...
    'This does not look like the Fig. 8 source JSON.');
rows = D.confirmation_all_12_grid_rows;
assert(numel(rows)==12,'Expected 12 archived confirmation rows.');

% Panel (b). LOCKED A12 ONLY, with exact recorded mean and sample SD.
tags = {'b1_longcp','b2_longcp','b2_shortcp'};
names = {'BPSK / CP16','QPSK / CP16','QPSK / CP4'};
bMu = nan(3,1); bSD = nan(3,1); bSeeds = nan(3,3);
for j = 1:3
    idx = find(strcmp({rows.tag},tags{j}) & strcmp({rows.variant},'back_out_a12'));
    assert(numel(idx)==1,'Expected exactly one A12 row for %s.',tags{j});
    r = rows(idx);
    raw = r.raw_seed_deltas;
    if ischar(raw) || isstring(raw)
        raw = sscanf(strrep(strrep(strrep(char(raw),'[',''),']',''),',',' '),'%f');
    end
    raw = double(raw(:));
    assert(numel(raw)==3 && all(isfinite(raw)),'Expected three A12 seed differences.');
    val = str2double(string(r.mean_delta_ber));
    spread = str2double(string(r.sample_sd));
    assert(isfinite(val) && isfinite(spread) && spread>=0);
    assert(abs(mean(raw)-val)<1e-10 && abs(std(raw,0)-spread)<1e-10, ...
        'Seed statistics do not match archived result for %s.',tags{j});
    bMu(j) = 1e3*val;
    bSD(j) = 1e3*spread;
    bSeeds(j,:) = (1e3*raw)';
end

% Panel (c). EIGHT pairs: 2 channels x 4 measured SNR points.
conditions = D.all_8_c_contrasts;
assert(numel(conditions)==8,'Expected 8 native-C transfer conditions.');
snrGrid = [12 16 20 24];
chNames = {'EVA','ETU'};
cVal = nan(2,4); cLow = nan(2,4); cHigh = nan(2,4);
for k = 1:numel(conditions)
    record = conditions(k);
    parts = regexp(char(record.condition),'^b1_longcp_seed1_(EVA|ETU)_(\d+)dB$', ...
        'tokens','once');
    assert(~isempty(parts),'Unexpected C condition: %s',record.condition);
    c = find(strcmp(chNames,parts{1}));
    s = find(snrGrid==str2double(parts{2}));
    assert(numel(c)==1 && numel(s)==1 && isnan(cVal(c,s)), ...
        'Duplicate/unexpected C condition: %s',record.condition);
    pair = record.paired;
    y = double(pair.candidate_minus_reference_ber);
    ci = double(pair.conditional_slot_bootstrap_95pct_ci(:));
    assert(numel(ci)==2 && all(isfinite(ci)) && ci(1)<=y && y<=ci(2));
    assert(pair.slots==2000 && pair.bits_per_slot==320 && pair.bootstrap_reps==2000, ...
        'Unexpected transfer interval setup.');
    assert(abs(y-(double(pair.candidate_ber)-double(pair.reference_ber)))<1e-12, ...
        'C BER contrast does not match error counts.');
    cVal(c,s)=1e3*y; cLow(c,s)=1e3*ci(1); cHigh(c,s)=1e3*ci(2);
end
assert(all(isfinite(cVal(:))) && all(isfinite(cLow(:))) && all(isfinite(cHigh(:))), ...
    'One or more transfer conditions missing.');

% Consistent restrained palette (navy, copper, teal and gray).
navy     = [0.12 0.26 0.39];
teal     = [0.05 0.48 0.48];
copper   = [0.79 0.33 0.17];
seedInk  = [0.35 0.43 0.52];
muted    = [0.38 0.44 0.50];
gridC    = [0.87 0.89 0.91];
zeroC    = [0.56 0.61 0.65];
lightCu  = [0.996 0.966 0.948];
lightNv  = [0.971 0.980 0.990];

figW=18.3; figH=12.8; % Full double-column width.
fig = figure('Name','Fig. 8 / Output-only A12 (v3)', ...
    'Units','centimeters','Position',[2.0 1.5 figW figH], ...
    'PaperUnits','centimeters','PaperSize',[figW figH], ...
    'PaperPosition',[0 0 figW figH], ...
    'Renderer','painters','InvertHardcopy','off','Color','w');

%% (a) Isolated output boundary: operator chain (schematic only).
a0 = axes('Parent',fig,'Units','centimeters', ...
    'Position',[0.84 9.76 16.65 2.14]);
hold(a0,'on');
set(a0,'Visible','off','XLim',[0 16.65],'YLim',[0 2.14]);
text(a0,0,2.12,'(a) Output-only A12: one changed boundary', ...
    'HorizontalAlignment','left','VerticalAlignment','bottom', ...
    'FontName','Arial','FontSize',10.1,'FontWeight','bold','Color',navy);

boxX = [0.28 4.27 8.23 12.60];
boxW = [3.02 3.04 3.53 3.05];
boxLabels = {{'Complex MAC','W8 x A8'}, ...
             {'Four INT32','reductions'}, ...
             {'Output quantizer','A12 / INT16 buffer'}, ...
             {'IFFT','FP32'}};
outline = {navy,navy,copper,navy};
fill = {lightNv,lightNv,lightCu,lightNv};
for j=1:4
    rectangle(a0,'Position',[boxX(j) 0.77 boxW(j) 0.93], ...
        'EdgeColor',outline{j},'FaceColor',fill{j},'LineWidth',0.96);
    text(a0,boxX(j)+boxW(j)/2,1.23,boxLabels{j}, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName','Arial','FontSize',9.1,'FontWeight','normal', ...
        'Color',outline{j});
    if j<4
        x1=boxX(j)+boxW(j)+0.08;
        x2=boxX(j+1)-0.12;
        plot(a0,[x1 x2],[1.23 1.23],'-','Color',navy,'LineWidth',0.9);
        patch(a0,[x2 x2-.14 x2-.14],[1.23 1.29 1.17],navy, ...
            'EdgeColor','none');
    end
end
text(a0,8.30,0.19, ...
    'Only the back-mixing output resolution changes; weights, input indices and FP32 IFFT stay fixed.', ...
    'Color',muted,'FontName','Arial','FontSize',8.0, ...
    'HorizontalAlignment','center','Interpreter','none');

%% (b) Confirmation: three configurations, 3 seeds + mean/sample SD.
bax = axes('Parent',fig,'Units','centimeters', ...
    'Position',[3.19 3.50 5.12 5.08]);
hold(bax,'on');
set(bax,'Color','w','Box','off','Layer','top', ...
    'FontName','Arial','FontSize',8.9,'LineWidth',0.70, ...
    'XColor',navy,'YColor',navy,'TickDir','out','YDir','reverse', ...
    'TickLabelInterpreter','none');
xlim(bax,[-2.48 0.22]); ylim(bax,[0.52 3.50]);
set(bax,'XTick',[-2.0 -1.5 -1.0 -0.5 0], ...
    'YTick',1:3,'YTickLabel',names);
title(bax,'(b) Independent confirmation', ...
    'FontName','Arial','FontSize',10.0,'FontWeight','bold','Color',navy);
for xv=[-2.0 -1.5 -1.0 -0.5]
    line(bax,[xv xv],[.52 3.50],'Color',gridC, ...
        'LineWidth',0.55,'LineStyle',':','HandleVisibility','off');
end
line(bax,[0 0],[.52 3.50],'Color',zeroC,'LineStyle','--', ...
    'LineWidth',1.05,'HandleVisibility','off');
for j=1:3
    line(bax,[-2.48 0.22],[j+.48 j+.48],'Color',gridC, ...
        'LineStyle',':','LineWidth',0.55,'HandleVisibility','off');
    dy=[-0.19 0 0.19]; shape={'o','s','^'};
    for k=1:3
        plot(bax,bSeeds(j,k),j+dy(k),shape{k},'LineStyle','none', ...
            'Color',seedInk,'MarkerFaceColor','w', ...
            'MarkerSize',5.7,'LineWidth',0.9,'HandleVisibility','off');
    end
    y=j+0.31; m=bMu(j); s=bSD(j);
    plot(bax,[m-s m+s],[y y],'-','Color',copper, ...
        'LineWidth',1.4,'HandleVisibility','off');
    plot(bax,[m-s m-s],[y-.055 y+.055],'-','Color',copper, ...
        'LineWidth',0.95,'HandleVisibility','off');
    plot(bax,[m+s m+s],[y-.055 y+.055],'-','Color',copper, ...
        'LineWidth',0.95,'HandleVisibility','off');
    plot(bax,m,y,'d','Color',copper,'MarkerFaceColor',copper, ...
        'MarkerSize',6.0,'LineWidth',0.95,'HandleVisibility','off');
end
xlabel(bax,'A12 - FFT W8A8 (x 10^{-3} BER)', ...
    'FontName','Arial','FontSize',8.5,'Color',navy);

%% (c) Native-C transfer: four true SNR locations per channel, paired CI.
% MATLAB line() uses property-value syntax for LineStyle; no positional ':'/'--'.
cax = axes('Parent',fig,'Units','centimeters', ...
    'Position',[10.81 3.50 6.24 5.08]);
hold(cax,'on');
set(cax,'Color','w','Box','off','Layer','top', ...
    'FontName','Arial','FontSize',8.9,'LineWidth',0.70, ...
    'XColor',navy,'YColor',navy,'TickDir','out');
xlim(cax,[11.1 24.9]); ylim(cax,[-0.84 0.10]);
set(cax,'XTick',snrGrid,'YTick',[-.8 -.6 -.4 -.2 0]);
title(cax,'(c) Native-C transfer', ...
    'FontName','Arial','FontSize',9.8,'FontWeight','bold','Color',navy);
for yy=[-.8 -.6 -.4 -.2]
    line(cax,[11.1 24.9],[yy yy],'LineStyle',':','Color',gridC,'LineWidth',0.50, ...
        'HandleVisibility','off');
end
line(cax,[11.1 24.9],[0 0],'LineStyle','--','Color',zeroC, ...
    'LineWidth',1.0,'HandleVisibility','off');
% Draw confidence intervals with basic line/plot primitives instead of
% ERRORBAR.  This avoids graphics-property / input-dispatch compatibility
% problems on some MATLAB R2021b installations, and makes the eight C
% contrasts completely explicit.  The marker is at the archived mean,
% whiskers span the archived conditional 95% paired-slot bootstrap CI.
lineColors = {navy,teal};
symbols    = {'o','s'};
hSeries    = gobjects(2,1);
for c = 1:2
    % Connecting lines join only the four measured SNR positions.
    hSeries(c) = plot(cax,snrGrid,cVal(c,:),'-', ...
        'Color',lineColors{c},'LineWidth',1.10, ...
        'HandleVisibility','on','DisplayName',chNames{c});
    for s = 1:numel(snrGrid)
        x = snrGrid(s);
        y = cVal(c,s);
        yLow = cLow(c,s);
        yHigh = cHigh(c,s);
        assert(isfinite(x) && isfinite(y) && isfinite(yLow) ...
            && isfinite(yHigh) && yLow <= y && y <= yHigh, ...
            'Invalid archived confidence interval for %s at %d dB.',chNames{c},x);
        % Vertical whisker plus two horizontal caps (x-units: dB).
        line(cax,[x x],[yLow yHigh],'Color',lineColors{c}, ...
            'LineWidth',0.95,'HandleVisibility','off');
        line(cax,[x-0.17 x+0.17],[yLow yLow],'Color',lineColors{c}, ...
            'LineWidth',0.95,'HandleVisibility','off');
        line(cax,[x-0.17 x+0.17],[yHigh yHigh],'Color',lineColors{c}, ...
            'LineWidth',0.95,'HandleVisibility','off');
        plot(cax,x,y,symbols{c},'LineStyle','none', ...
            'Color',lineColors{c},'MarkerFaceColor','w', ...
            'MarkerSize',6.0,'LineWidth',1.10,'HandleVisibility','off');
    end
end
xlabel(cax,'SNR (dB)','FontName','Arial','FontSize',9.1,'Color',navy);
ylabel(cax,'C A12 - C W8A8 (x 10^{-3} BER)', ...
    'FontName','Arial','FontSize',8.3,'Color',navy);
text(cax,0.98,0.92,'BPSK / CP16, seed 1', ...
    'Units','normalized','HorizontalAlignment','right', ...
    'FontName','Arial','FontSize',8.1,'Color',muted);
legend(cax,hSeries,chNames,'Location','southwest','Box','off', ...
    'FontName','Arial','FontSize',8.2);

%% Legends and scope notes are below the charts, not inside data regions.
legAx = axes('Parent',fig,'Units','centimeters', ...
    'Position',[1.18 2.15 8.2 0.49],'Visible','off');
hold(legAx,'on'); xlim(legAx,[0 8.2]); ylim(legAx,[0 1]);
sx=[.11 1.87 3.63 5.38]; shapes={'o','s','^'};
labels={'Seed 1','Seed 2','Seed 3','Mean +/- sample SD'};
for k=1:4
    if k<=3
        plot(legAx,sx(k),.5,shapes{k},'LineStyle','none', ...
            'Color',seedInk,'MarkerFaceColor','w', ...
            'MarkerSize',5.6,'LineWidth',0.9);
    else
        plot(legAx,[sx(k)-.19 sx(k)+.19],[.5 .5],'-', ...
            'Color',copper,'LineWidth',1.3);
        plot(legAx,sx(k),.5,'d','LineStyle','none', ...
            'Color',copper,'MarkerFaceColor',copper,'MarkerSize',6.0);
    end
    text(legAx,sx(k)+.26,.5,labels{k}, ...
        'FontName','Arial','FontSize',8.0,'Color',navy, ...
        'VerticalAlignment','middle','Interpreter','none');
end
annotation(fig,'textbox',[.56 .166 .41 .043], ...
    'String','Whiskers: conditional 95% paired-slot bootstrap CI (2,000 draws)', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',7.5,'Color',muted,'Interpreter','none');
annotation(fig,'textbox',[.052 .058 .91 .072], ...
    'String',{'Negative = fewer errors. (b) Independent confirmation; sample SD across 3 seeds.', ...
              '(c) Reused BPSK/CP16 seed-1 subset; conditional intervals, not independent replication.'}, ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',8.0,'Color',muted,'Interpreter','none');

%% Save MATLAB-native file plus editable vector/raster paper output.
outputBase=fullfile(rootDir,'fig8_matlab_publication_v3');
savefig(fig,[outputBase '.fig']);
print(fig,[outputBase '.pdf'],'-dpdf','-painters');
print(fig,[outputBase '.svg'],'-dsvg','-painters');
print(fig,[outputBase '.png'],'-dpng','-r600');
fprintf('Fig. 8 saved: %s.[fig/pdf/svg/png]\n',outputBase);
fprintf('Validated 3 A12 confirmation summaries x 3 seeds; 8 C contrasts and recorded paired CIs.\n');
fprintf('No BER recomputation, smoothing, or changes to experimental points.\n');
