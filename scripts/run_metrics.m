f_a = fopen(fn_a, 'r');
v_a = fread(f_a, 'uint32');
v_a = reshape(v_a, [1024 1024 25]);
fclose(f_a);

f_b = fopen(fn_b, 'r');
v_b = fread(f_b, 'uint32');
v_b = reshape(v_b, [1024 1024 25]);
fclose(f_b)

metrics(v_a, v_b);
