%script to perform nuclei segmentation and feature extraction on the cluster

close all; clear all; clc;

%parameters
Desired = 20; %objective magnification
T = 4096; %tilesize
InputFolder = '/data2/Images/bcrTCGA/diagnostic_block_HE_section_image/';
OutputFolder = '/bigdata3/mnalisn_scratch/missing/';
Mem = '512'; %free memory
Prefix = 'Nuclei-';

%check if output folder exists
if(exist(OutputFolder, 'dir') ~= 7)
    mkdir(OutputFolder);
end

%get list of .svs files in integen.org folders
Folders = dir([InputFolder 'intgen.org_GBM*']); 
Folders(~[Folders(:).isdir]) = [];
Folders = {Folders(:).name};
Files = cell(1, length(Folders));
Slides = cell(1,length(Folders));
for i = 1:length(Folders)
    Files{i} = dir([InputFolder Folders{i} '/*.svs']);
    Files{i} = {Files{i}(:).name};
    Slides{i} = cellfun(@(x)x(1:end-4), Files{i}, 'UniformOutput', false);
    Files{i} = cellfun(@(x) [InputFolder Folders{i} '/' x], Files{i}, 'UniformOutput', false);
end
Files = [Files{:}].';
Slides = [Slides{:}];

%iterate through slides, submitting one job per slide
for i = 1:length(Files)
    
    %check if output folder exists
    if(exist([OutputFolder Slides{i}], 'dir') ~= 7)
        mkdir([OutputFolder Slides{i}]);
    end
    
    %generate job string
    Job = sprintf('matlab -nojvm -nodesktop -nosplash -logfile "%s" -r "SegmentationProcessTiles(''%s'', %g, %g, ''%s''); exit;"',...
                    [OutputFolder Prefix num2str(i) '.txt'],...
                    Files{i}, Desired, T,...
                    [OutputFolder Slides{i} '/']);
    
    %submit job
    [status, result] = SubmitPBSJob(Job, [Prefix num2str(i)], Mem);
    
    %update console
    fprintf('job %d, folder: %s, status: %s.', i, Files{i}, result);
    
end