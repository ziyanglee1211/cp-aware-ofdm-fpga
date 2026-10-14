%% Figure 6: corrected-PDP component interventions (MATLAB R2021b)
% Source: fig_component_evidence_source.json
% Figures: (a) EVA threshold contrasts, (b) ETU threshold contrasts,
%          (c) all measured CP-folding BER contrasts, (d) high-SNR detail.
% Preserves all 3 paired seeds and original sample SD (not 95% CI).
% Identity cases are right-censored (>28 dB), NEVER imputed.
% No smoothing or re-estimation of thresholds is performed.
%
% Put this file and fig_component_evidence_source.json in the same folder.
% Run in MATLAB command window: plot_fig6_component_matlab

clearvars; close all; clc;
thisFile = mfilename('fullpath');
if isempty(thisFile)
    rootDir = pwd;
else
    rootDir = fileparts(thisFile);
end
jsonPath = fullfile(rootDir, 'fig_component_evidence_source.json');
assert(exist(jsonPath,'file') == 2, 'Missing source JSON: %s',jsonPath);
J = jsondecode(fileread(jsonPath));
assert(isfield(J,'data') && isfield(J.data,'EVA') && isfield(J.data,'ETU'), ...
    'The supplied JSON does not match Fig. 6 component evidence.');

% Editorial palette: blue fixed/reference, orange architecture contrast,
% teal companion channel. Marker shapes, not colors alone, encode seeds.
navy = [0.10,0.24,0.42];
copper = [0.78,0.32,0.14];
teal = [0.02,0.49,0.48];
gray = [0.38,0.41,0.45];
light = [0.85,0.87,0.89];

% Final full-column figure; coordinate positions fixed for reliable print.
fw = 18.0; fh = 14.0; % centimeters
fig = figure('Color','w','Units','centimeters','Position',[2 2 fw fh], ...
    'PaperUnits','centimeters','PaperSize',[fw fh], ...
    'PaperPosition',[0 0 fw fh], 'Renderer','painters','Name','Fig. 6 / Component effects');

names = {'Two-block control','Remove CP folding','Serial two-block', ...
         'Six-block differential','Fixed RE gain','Fixed front identity','Fixed back identity'};
keys  = {'slim2','discard_cp','serial2','full6','fixed_basic_gain','fixed_front','fixed_back'};

% --- First row: paired 2%-BER threshold effects, separate x-axis ranges ---
axA = axes('Parent',fig,'Units','centimeters','Position',[4.35 8.40 4.70 4.10]);
axB = axes('Parent',fig,'Units','centimeters','Position',[11.60 8.40 5.55 4.10]);
paint_threshold(axA, J.data.EVA, keys, names, [-0.12 0.68], navy, copper, gray, light, true);
paint_threshold(axB, J.data.ETU, keys, names, [-0.22 3.65], navy, copper, gray, light, false);
title(axA,'(a) EVA — expanded scale','FontSize',9,'FontWeight','bold');
title(axB,'(b) ETU','FontSize',9,'FontWeight','bold');
xlabel(axA,'2%-BER threshold shift (dB)','FontSize',8);
xlabel(axB,'2%-BER threshold shift (dB)','FontSize',8);

% --- Shared key in reserved gap; use actual plotted glyphs, not Unicode ---
keyAx = axes('Parent',fig,'Units','centimeters','Position',[3.55 7.28 13.2 0.55], ...
    'Visible','off','Color','none');
hold(keyAx,'on'); xlim(keyAx,[0 9.8]); ylim(keyAx,[0 1]);
markerTypes = {'o','s','^','d'};
markerXs = [0.3,2.3,4.3,6.3];
markerTxt = {'Seed 1','Seed 2','Seed 3','Mean \pm sample SD'};
for k = 1:4
    if k<4
        plot(keyAx,markerXs(k),0.5,markerTypes{k},'Color',gray, ...
            'MarkerFaceColor','w','MarkerSize',5.3,'LineWidth',0.8);
    else
        plot(keyAx,[6.12,6.48],[0.5,0.5],'-','Color',copper,'LineWidth',1.15);
        plot(keyAx,markerXs(k),0.5,'d','Color',copper, ...
            'MarkerFaceColor',copper,'MarkerSize',5.3);
    end
    text(keyAx,markerXs(k)+0.22,0.5,markerTxt{k},'FontSize',7.6, ...
        'VerticalAlignment','middle','Color',navy,'Interpreter','tex');
end

% --- Second row: same measured CP data, full range and detail. ---
axC = axes('Parent',fig,'Units','centimeters','Position',[1.70 1.33 6.95 4.20]);
axD = axes('Parent',fig,'Units','centimeters','Position',[10.3 1.33 6.88 4.20]);

snrAll = 0:4:28;
[evM,evS] = extract_cp_differences(J.data.EVA.discard_cp.at_snr,snrAll);
[etM,etS] = extract_cp_differences(J.data.ETU.discard_cp.at_snr,snrAll);
% Multiply by 1e3 for legibility. This is a linear display-unit change only.
plot_cp_panel(axC,snrAll,evM*1000,evS*1000,etM*1000,etS*1000,navy,teal,light);
plot_cp_panel(axD,snrAll(5:end),evM(5:end)*1000,evS(5:end)*1000,...
    etM(5:end)*1000,etS(5:end)*1000,navy,teal,light);
set(axC,'XLim',[-0.5 28.5],'YLim',[-1.1 12.4], 'XTick',0:4:28);
set(axD,'XLim',[15.5 28.5],'YLim',[-0.67 1.10], 'XTick',16:4:28);
title(axC,'(c) CP folding: full SNR grid','FontSize',9,'FontWeight','bold');
title(axD,'(d) High-SNR detail (same data)','FontSize',9,'FontWeight','bold');
xlabel(axC,'SNR (dB)','FontSize',8);
xlabel(axD,'SNR (dB)','FontSize',8);
ylabel(axC,'BER change (x 10^{-3})','FontSize',8);
ylabel(axD,'BER change (x 10^{-3})','FontSize',8);
% Channel names do not require another large floating legend.
text(axC,0.965,0.91,'EVA','Units','normalized','HorizontalAlignment','right', ...
    'FontSize',7.9,'Color',navy,'FontWeight','bold');
text(axC,0.965,0.81,'ETU','Units','normalized','HorizontalAlignment','right', ...
    'FontSize',7.9,'Color',teal,'FontWeight','bold');

% Export vector figures at exact printed dimensions, plus 600-dpi proof PNG.
rootName = fullfile(rootDir,'fig6_matlab_publication');
set(fig,'InvertHardcopy','off');
print(fig,[rootName '.pdf'],'-dpdf','-painters');
print(fig,[rootName '.svg'],'-dsvg','-painters');
print(fig,[rootName '.png'],'-dpng','-r600');
fprintf('\nExported Figure 6 to:\n  %s.pdf\n  %s.svg\n  %s.png\n',rootName,rootName,rootName);
fprintf('Statistical meaning: 3 paired seeds; diamond=mean, whisker=sample SD.\n');
fprintf('Censored front/back identity variants shown only as >28 dB.\n');

%% Local functions (MATLAB R2021b supports functions at end of scripts)
function paint_threshold(ax, cohort, keys, names, xr, navy, copper, gray, light, showLabels)
    hold(ax,'on');
    ax.FontName = 'Arial'; ax.FontSize = 7.6; ax.LineWidth = 0.7;
    ax.TickDir = 'out'; ax.Box = 'off'; ax.Layer = 'top';
    ax.YDir = 'reverse';
    ylim(ax,[0.4 7.7]); xlim(ax,xr);
    yticks(ax,1:7); % MATLAB requires ascending tick values
    if showLabels
        yticklabels(ax,names);
    else
        yticklabels(ax,{});
    end
    % A dedicated no-imputation zero-reference, not a fitted trend.
    line(ax,[0 0],[0.4 7.7],'Color',[0.58 0.60 0.62], ...
        'LineStyle','--','LineWidth',0.75,'HandleVisibility','off');
    for j = 1:numel(keys)
        s = cohort.(keys{j}).paired_variant_minus_slim2_db;
        if isempty(s.mean) || ~isfinite(s.mean) || any(~isfinite(s.per_seed))
            % Right-censored thresholds: keep absolute >28 dB, not delta.
            if (j==6 || j==7)
                xx = xr(1)+0.50*(xr(2)-xr(1));
                text(ax,xx,j,'>28 dB (3/3 seeds)','FontSize',7.1, ...
                    'FontAngle','italic','Color',gray,'HorizontalAlignment','center');
            end
            continue;
        end
        v=double(s.per_seed(:)); m=double(s.mean); sd=double(s.sample_sd);
        assert(numel(v)==3 && abs(m-mean(v))<1e-9, ...
            'Unexpected three-seed summary at %s.',keys{j});
        if abs(sd-std(v,0))>1e-9
            error('Sample SD mismatch for %s.',keys{j});
        end
        offs = [-0.19 -0.06 0.07];
        mark = {'o','s','^'};
        for k=1:3
            plot(ax,v(k),j+offs(k),mark{k},'MarkerSize',4.8, ...
                'MarkerFaceColor','w','Color',gray,'LineWidth',0.78);
        end
        % One horizontal ±sample-SD whisker and mean diamond per arm.
        ym = j + 0.21;
        line(ax,[m-sd m+sd],[ym ym],'Color',copper,'LineWidth',1.25);
        cap = 0.063;
        line(ax,[m-sd m-sd],[ym-cap ym+cap],'Color',copper,'LineWidth',1.05);
        line(ax,[m+sd m+sd],[ym-cap ym+cap],'Color',copper,'LineWidth',1.05);
        plot(ax,m,ym,'d','Color',copper,'MarkerFaceColor',copper, ...
            'MarkerSize',5.1,'LineWidth',0.85);
    end
    xtickformat(ax,'%.1f');
    for yy = 1.5:1:6.5
        line(ax,xr,[yy yy],'Color',light,'LineStyle',':', ...
            'LineWidth',0.4,'HandleVisibility','off');
    end
end

function [avg,spread] = extract_cp_differences(atSNR,snr)
    avg=zeros(size(snr)); spread=zeros(size(snr));
    fn = fieldnames(atSNR);
    for k=1:numel(snr)
        key = matlab.lang.makeValidName(sprintf('%.1f',snr(k)));
        if ~isfield(atSNR,key)
            % Fallback if JSON parser used a different legal key convention.
            hit = find(strcmp(fn,key),1);
            assert(~isempty(hit),'Cannot find measured SNR %g in JSON.',snr(k));
            key = fn{hit};
        end
        z = atSNR.(key).paired_variant_minus_slim2_ber;
        avg(k) = double(z.mean);
        spread(k) = double(z.sample_sd);
        v = double(z.per_seed(:));
        assert(numel(v)==3 && abs(avg(k)-mean(v))<1e-10 && ...
            abs(spread(k)-std(v,0))<1e-10, 'CP seed data disagree at %g dB.',snr(k));
    end
end

function plot_cp_panel(ax,x,mEva,sEva,mEtu,sEtu,navy,teal,light)
    hold(ax,'on'); ax.Color='w'; ax.FontName='Arial'; ax.FontSize=7.6;
    ax.LineWidth=0.7; ax.Box='off'; ax.TickDir='out'; ax.Layer='top';
    line(ax,[min(x) max(x)],[0 0],'Color',light,'LineStyle','--','LineWidth',0.85);
    errorbar(ax,x,mEva,sEva,'-o','Color',navy,'MarkerFaceColor','w', ...
        'LineWidth',1.15,'MarkerSize',4.4,'CapSize',4);
    errorbar(ax,x,mEtu,sEtu,'-s','Color',teal,'MarkerFaceColor','w', ...
        'LineWidth',1.10,'MarkerSize',4.2,'CapSize',4);
end
