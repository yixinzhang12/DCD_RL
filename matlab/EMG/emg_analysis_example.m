% Assume variable 'emg' is one channel (i.e., one muscle) from your emg
data = readtable('trial1.txt');
data = data(:,{'CH1', 'CH2', 'CH3', 'CH4'});

% data

emg = data(:,"CH1");
Fs = 2000; % replace with your own sampling frequency
%% COMMON WORKFLOW #1:
%STEP 1 = remove offset
offset = repmat(mean(emg), size(emg, 1),1);
emg_off = emg - offset;
%STEP 2 = filtering
[b0,a0] = butter(2,[59 61]/(Fs/2),'stop'); % Band Stop at 60Hz; 4th order is a common option as well; depending on raw data quality may not need it
[b,a] = butter(2,[30 450]/(Fs/2),'bandpass'); % Bandpass 30-450Hz; 4th order is a common option as well
emg_filt_stop = filtfilt(b0,a0,emg_off);
emg_filt_pass = filtfilt(b,a,emg_filt_stop);
%STEP 3 = rectification (always use rectified EMG to compute EMG amplitude)
emg_rect = abs(emg_filt_pass);
%STEP 4 = smoothing 
% e.g., using a moving average window

%% COMMON WORKFLOW #2:
%STEP 1 = remove offset
offset = repmat(mean(emg), length(emg),1);
emg_off = emg - offset;
%STEP 2 = filtering
[b0,a0] = butter(2,[59 61]/(Fs/2),'stop'); % Band Stop at 60Hz; 4th order is a common option as well; depending on raw data quality may not need it
[b,a] = butter(2,[30 450]/(Fs/2),'bandpass'); % Bandpass 30-450Hz; 4th order is a common option as well
emg_filt_stop = filtfilt(b0,a0,emg_off);
emg_filt_pass = filtfilt(b,a,emg_filt_stop);
%STEP 3 = rectification + smoothing
% instead of using abs(), directly do a moving RMS (see example in the
% section below)
% NOTE: once get filtered EMG, it's either [abs() + moving average] OR [moving RMS]; you shouldn't do RMS
% on top of rectified EMG

%% code to extract MVC value from a script I used for another project
% ORGANIZE, OFFSET, FILTER, AND ENVELOPE EMG DATA, INCL. TRIG
% idx = [1 1 3 3 5 5 7 9 11 13 15 15; 2 2 4 4 6 6 8 10 12 14 16 16]; %SCI1
idx = [1 1 3 3 5 5 7 9 11 11 13 13; 2 2 4 4 6 6 8 10 12 12 14 14]; %SCI3
% idx = [1 1 3 3 5 7 9 11 13 15 17 17; 2 2 4 4 6 8 10 12 14 16 18 18]; % AB
mvc_titles = {'LRA', 'RRA', 'LES', 'RES', 'LGMax', 'RGMax', 'LGMed', 'RGMed', 'LSOL', 'RSOL', 'LPFM', 'RPFM'};
emgf = 2000;
[b0,a0] = butter(4, [59 61]/(emgf/2), 'stop'); %notch at 60Hz
[b,a] = butter(4,[50 450]/(emgf/2),'bandpass'); % bandpass 50-450Hz
Len = 100; % 50ms moving rms window
Overlap = Len-1;
MovRMS = dsp.MovingRMS(Len,Overlap);

for i = 1:2 %2 trials per muscle
    for k = 1:nRightChanMVC
        mvc2use(i).data(k).trig = mvc(idx(i,k)).all(:,trigCol(1));
        mvc2use(i).data(k).emgraw = mvc(idx(i,k)).emgraw(:,k);

        offset = repmat(mean(mvc2use(i).data(k).emgraw), length(mvc2use(i).data(k).emgraw), 1);
        mvc2use(i).data(k).emgoff = mvc2use(i).data(k).emgraw - offset;

        mvc2use(i).data(k).emgnotch = filtfilt(b0,a0,mvc2use(i).data(k).emgoff);
        mvc2use(i).data(k).emgfilt = filtfilt(b,a,mvc2use(i).data(k).emgnotch);
        mvc2use(i).data(k).emgrms  = MovRMS(mvc2use(i).data(k).emgfilt);
        mvc2use(i).data(k).emgrms(1:Len/2) = []; %account for 1/2 wsz shift ('delay') in moving rms algorithm

        % this is the section to visualize EMG with trigger/event marker
        % overlaid, and you can click on the middle to select a 1-sec
        % window to compute EMG amplitude
        figure(i)
        hold off
        yyaxis left
        plot(mvc2use(i).data(k).emgfilt);
        hold on
        plot(mvc2use(i).data(k).emgrms, 'linewidth', 1.5);
        yyaxis right
        plot(mvc2use(i).data(k).trig, 'r'); %comment out for AB due to trig dysfunction
        title(mvc_titles(k));
        set(gcf, 'WindowState', 'maximized');
        [x(i,k),~] = ginput(1);
        mvc2use(i).data(k).avg = mean(mvc2use(i).data(k).emgrms(x(i,k)-emgf/2:x(i,k)+emgf/2));
        close all
    end
end

mvc_values = zeros(nRightChanMVC,1);
for k = 1:nRightChanMVC-1
    mvc_values(k,1) = mean([mvc2use(1).data(k).avg ...
        mvc2use(2).data(k).avg]);
end