% -------------------------------------------------------------------------
% Генератор синтетичних ЕПР-спектрів з лінією Дайсона з аугментацією
%
% Цей скрипт створює набір синтетичних спектрів ЕПР у вигляді похідної
% сигналу лінії Дайсона з різними параметрами.
%
% Для кожного спектра випадково генеруються параметри:
%   - B0 : положення резонансного поля (G)
%   - dB : ширина лінії (peak-to-peak, G)
%   - p  : параметр асиметрії (визначає форму лінії Дайсона)
%   - I  : амплітуда сигналу (фіксована)
%
% Додатково до кожного спектра додається білий шум (% від амплітуди),
% зміщення базової лінії та нахил базової лінії
%
% Результати зберігаються у двох масивах:
%   - X : [N x 4096] — матриця спектрів (4096 точок у кожному спектрі)
%   - y : [N x 4]    — відповідні параметри [B0, dB, p, I]
%
% Дані експортуються у форматі .npy (потрібна бібліотека npy-matlab).
% https://github.com/kwikteam/npy-matlab
% -------------------------------------------------------------------------
% addpath('npy-matlab'); % Шлях до бібліотеки з функціями npy
clear, clf

% Параметри генерації
numSamples = 3000;
numPoints = 4096;
B1 = 0;
B2 = 6600;
f = 9.4;  % ГГц

% Ініціалізація масивів
X = zeros(numSamples, numPoints);
y = zeros(numSamples, 4); % [B0, dB, p, I]

% Масив магнітного поля
B = linspace(B1, B2, numPoints);

for i = 1:numSamples
    % Випадкові параметри
    B0 = 3300 + rand() * 100; %B0 від 330 до 340 мТ
    
    % Для  dB > 100 - інша формула, перерахувати!
    dB = 1 + rand() * (1000 - 1); %dB від 1 Гс до 1000 Гс
    
    p = 0.01 + rand() * (2.5 - 0.01);
    %p = 1 + rand() * (2 - 1);
    %p = 0.0001 + rand() * (20 - 0.0001); %p = 0.0001 до 20 TOO MUCH
    %I = (exp(randn() * 0.5) - 0.2) / 4.3;
    %I = max(min(I, 1), 0.25);  % жорстке обрізання
    %I = 0.5 + 0.5 * rand();  % або:
    %I = exp(randn() * 0.5); % логнормально
    I = 1 ; % 5..6
    %I = 1e5 + rand() * (1e6 - 1e5);

    % Обчислення профілю
    x = 2 * (B - B0) / (sqrt(3) * dB);
    denom = 2 * p * (cosh(p) + cos(p));
    A = (sinh(p) + sin(p)) / denom + (1 + cosh(p) * cos(p)) / (cosh(p) + cos(p))^2;
    D = (sinh(p) - sin(p)) / denom + (sinh(p) * sin(p)) / (cosh(p) + cos(p))^2;

    dI = I * ((-A * 2 * x) ./ (1 + x.^2).^2 + (D * (1 - x.^2)) ./ (1 + x.^2).^2);

    % ------------------
    % АУГМЕНТАЦІЯ ДАНИХ:
    % ------------------

    signalAmplitude = max(dI) - min(dI); % Оцінка амплітуди сигналу
    desired_SNR = 750;
    noise_std = signalAmplitude / (2 * desired_SNR); % Розрахунок std шуму
    noise = noise_std * randn(1, numPoints);  % Генерація білого шуму   
    dI_noisy = dI + noise; % Додавання шуму до сигналу

    % Замість dI використовуємо dI_noisy у виводі/збереженні

    % 2. Зміщення базової лінії (±0.2% сигналу)
    baseline_shift = 0.002 * (2 * rand() - 1);
    dI_noisy = dI_noisy + baseline_shift;

    % 3. Нахил базової лінії (±0.1%)
    slope = 0.001 * (2 * rand() - 1);
    dI_noisy = dI_noisy + slope * linspace(-1, 1, numPoints);

    % Запис
    X(i, :) = dI_noisy;
    y(i, :) = [B0, dB, p, I];
end

% Збереження у форматі .npy
ThisFolder = fileparts(mfilename('fullpath'));  % шлях до цього .m-файлу
writeNPY(X, fullfile(ThisFolder, 'X_dyson.npy'));
writeNPY(y, fullfile(ThisFolder, 'y_dyson.npy'));

disp('Генерація завершена. Файли X_dyson.npy та y_dyson.npy збережено.');

% Вивід 3 випадкових спектрів
close all;
randomIdx = randperm(numSamples, 10);
for i = 1:10
    idx = randomIdx(i);
    figure;
    plot(B, X(idx, :), 'LineWidth', 1.2);
    title(sprintf('Синтетичний спектр #%d\nB0 = %.2f G, dB = %.2f G, p = %.2f, I = %.0f', ...
        idx, y(idx, 1), y(idx, 2), y(idx, 3), y(idx, 4)));
    xlabel('Магнітне поле (G)');
    ylabel('dI/dB (a.u.)');
    grid on;
end