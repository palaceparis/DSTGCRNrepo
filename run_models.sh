#!/bin/bash

# # Provide inputs to the first script
# echo -e "dryrun\n\n42" | bash run_FC-LSTM.sh

# # Provide inputs to the second script
# echo -e "dryrun\n" | bash run_MLP.sh
# echo -e "dryrun\n\n" | bash run_AGCRN_dynamic_laplacian.sh

# echo -e "dryrun\n\n" | bash run_AGCRN.sh

# echo -e "dryrun\n\n" | bash run_FC-LSTM.sh

# echo -e "dryrun\n\n" | bash run_MLP.sh

echo -e "dryrun\n\n" | bash run_AGCRN.sh "AGCRN_us_data_10_1"

echo -e "dryrun\n\n" | bash run_AGCRN_dynamic_laplacian.sh "DSTGCRN_us_data_10_1"




