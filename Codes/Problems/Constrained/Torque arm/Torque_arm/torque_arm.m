function [sreq, mass] = torque_arm(stot, fx, fy, draw)

    if nargin < 4
        draw = 0;
    end

    %% Validate inputs
    stot = double(stot);
    fx = double(fx(:));
    fy = double(fy(:));

    N = size(stot, 1);

    if size(stot, 2) ~= 7
        error('torque_arm:InvalidStot', ...
              'stot must be an N-by-7 matrix.');
    end

    if numel(fx) ~= N
        error('torque_arm:InvalidFx', ...
              'fx must contain one value for each row of stot.');
    end

    if numel(fy) ~= N
        error('torque_arm:InvalidFy', ...
              'fy must contain one value for each row of stot.');
    end

    %% Start or reuse parallel pool
    pool = gcp('nocreate');
    
    if isempty(pool)
        cluster = parcluster('Processes');
        pool = parpool(cluster);
    end

    %% Preallocate outputs
    sreq = zeros(N, 1);
    cr_a = zeros(N, 1);
    count = zeros(N, 1);

    %% Run analyses in parallel
    startTime = tic;

    parfor c = 1:N

        s = stot(c, :);
        Fx = fx(c);
        Fy = fy(c);

        [sreq(c), cr_a(c), count(c)] = ...
            stress_analysis(s, Fx, Fy, draw);
    end

    elapsedTime = toc(startTime);

    %% Calculate mass
    % cr_a: cm^2
    % area_m2: m^2
    area_m2 = cr_a * 1e-4;

    thickness = 0.003;   % metres
    density = 7850;      % kg/m^3

    mass = area_m2 * thickness * density;

    %% Display timing
    fprintf('Completed %d simulations in %.2f seconds.\n', ...
            N, elapsedTime);

    fprintf('Average rate: %.2f simulations/second.\n', ...
            N / elapsedTime);

end

%%
delete(gcp('nocreate'));

pool = parpool('Processes', 2);