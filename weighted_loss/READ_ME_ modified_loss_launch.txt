запуск модели с модифицированным лоссом:

python3 -i CODEX_gene_perturbation.py \
  -l 512 128 64 \
  -s 1 \
  -dn norman \
  --deg_weighting \
  --deg_weight_alpha 3.0 \
  --deg_weight_max 5.0 \
  --deg_effect_scale 4.0 \
  --deg_top_k 20

-l  -  количества нейронов в слоях
-dn  -  датасет
--deg_weighting  -  добавлять ли в loss взвешивание генов, учитывающее l2fc
--deg_weight_alpha  -  гиперпараметр - коэффициент для рассчета веса (я бы брал примерно равный --deg_effect_scale, либо 1)
--deg_weight_max  -  ограничитель максимального веса гена, чтобы не было слишком большого перекоса
--deg_effect_scale  -  максимальное значение l2fc, на которое будет нормироваться вес 
--deg_top_k  -  сколько top-DEG генов учитывать для каждого perturbation condition.
