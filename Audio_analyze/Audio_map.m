% 清除工作区并关闭已有图形
clear;
close all;
clc;

% 设置默认字体
set(groot, 'defaultAxesFontName', 'Arial');

%% 参数设置
fs = 16000;           % 采样率
duration = 3;         % 音频时长
n_samples = fs * duration;
nfft = 2048;          % FFT点数
window = hamming(512); % 窗函数
noverlap = 256;       % 重叠点数
mel_filters = 40;     % Mel滤波器数量

%% 文件路径配置
base_dir = 'E:\School_Work\PhD_1\Transformer-AVSE-V5\Test_Output\S00022';
file_names = {'target.wav', 'enhanced.wav', 'denoised.wav', 'mixed.wav'};
titles = {'Target', 'Enhanced', 'Denoised', 'Mixed'}; % 对应图例标签

%% 读取并预处理音频
audio_data = cell(1,4);
for i = 1:4
    file_path = fullfile(base_dir, file_names{i});
    
    % 检查文件是否存在
    if ~exist(file_path, 'file')
        error('文件不存在: %s', file_path);
    end
    
    % 读取音频
    [audio, fs_read] = audioread(file_path);
    
    % 验证采样率
    if fs_read ~= fs
        error('采样率不一致: %s (期望 %d Hz，实际 %d Hz)',...
              file_names{i}, fs, fs_read);
    end
    
    % 截断/补零到固定长度
    if length(audio) < n_samples
        audio(end+1:n_samples, :) = 0;
    else
        audio = audio(1:n_samples, :);
    end
    
    % 归一化并存储
    audio_data{i} = audio / max(abs(audio));
end

%% 计算频谱特征
% 预分配存储
mel_spectrograms = cell(1,4);
energy_spectrums = cell(1,4);
freq_axis = []; % 频率轴初始化

figure('Position', [100, 100, 1200, 800]); % 统一图形窗口

% 并行计算所有音频特征
for i = 1:4
    % 计算STFT
    [S, f, t] = spectrogram(audio_data{i}, window, noverlap, nfft, fs);
    
    % 计算Mel频谱
    [mel_spec, t_mel] = melSpectrogram(audio_data{i}, fs,...
        'Window', window,...
        'OverlapLength', noverlap,...
        'FFTLength', nfft,...
        'NumBands', mel_filters);
    
    % 存储结果
    mel_spectrograms{i} = mel_spec;
    energy_spectrums{i} = sum(abs(S).^2, 2);
    
    % 记录频率轴（只需一次）
    if isempty(freq_axis)
        freq_axis = f;
    end
    
    %% 绘制Mel频谱子图
    subplot(2, 2, i);
    imagesc(t_mel, 1:mel_filters, 10*log10(mel_spec));
    axis xy;
    colorbar;
    title(sprintf('%s Signal', titles{i}));
    xlabel('Time (s)');
    ylabel('Mel Band');
end
sgtitle('Mel Spectrogram Comparison', 'FontSize', 14);

%% 绘制能量对比图
figure('Position', [150, 150, 1000, 500]); % 第二个图形窗口
hold on;

% 配色方案
colors = lines(4);

% 绘制所有能量曲线
for i = 1:4
    plot(freq_axis, 10*log10(energy_spectrums{i}),...
        'LineWidth', 1.5,...
        'DisplayName', titles{i},...
        'Color', colors(i,:));
end

hold off;
grid on;
legend('Location', 'best');
title('Frequency Domain Energy Distribution');
xlabel('Frequency (Hz)');
ylabel('Power Spectrum Density (dB)');
xlim([0 fs/2]);
set(gca, 'FontSize', 10);