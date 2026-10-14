function export_native_figures(outputDir)
% Re-export stored objects; do not compute new BER or alter original inputs.
assert(nargin==1,'Provide a new output directory.');
assert(~isfolder(outputDir) && ~isfile(outputDir),'Do not overwrite an existing output.');
root=fileparts(fileparts(mfilename('fullpath')));
mkdir(outputDir);
for index=5:10
    source=fullfile(root,'figures','native',sprintf('Figure_%02d.fig',index));
    assert(isfile(source),'Missing original native figure.');
    fig=openfig(source,'invisible');
    cleanup=onCleanup(@() close(fig));
    print(fig,fullfile(outputDir,sprintf('Figure_%02d_reexport.pdf',index)),'-dpdf','-painters');
    print(fig,fullfile(outputDir,sprintf('Figure_%02d_reexport.png',index)),'-dpng','-r300');
    clear cleanup;
end
fprintf('Stored Fig.5-10 objects exported. This is not a new scientific experiment.\n');
end
