import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define the column width in inches
column_width_inches = 8
# Define the aspect ratio of the figure (optional)
aspect_ratio = 0.75

# Calculate the figure width and height based on the column width and aspect ratio
figure_width_inches = column_width_inches  
figure_height_inches = column_width_inches * aspect_ratio

# ticks_size = 12
# label_size = ticks_size*1.2


sns.set()
# sns.set_style('whitegrid')

plt.rcParams["font.family"] = "Times New Roman"
font = 20
bar_width = 0.2
scale_font=1
h_space=1
# Set the default text font size
# plt.rc('font', size=14, weight='bold')
# plt.rc('figure', figsize = (8,6))
plt.rc('figure', figsize = (figure_width_inches,figure_height_inches))

# plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rc('font', size=font)
# plt.rc('text', usetex=True)
# Set the axes title font size
# plt.rc('axes', titlesize=font)
# Set the axes labels font size
plt.rc('axes', labelsize=font)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=font)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=font)
# Set the legend font size
plt.rc('legend', fontsize=font*scale_font)
# Set the font size of the figure title
plt.rc('figure', titlesize=font)

TALD_gain_dict = {}
CKD_gain_dict = {}
pairs_order = []


def analysis(method_names, data, filename, title):
    algorithm1_data, algorithm2_data, algorithm3_data = np.array(data)
    # print(filename)
    TALD_gain = algorithm2_data - algorithm1_data
    coded_gain = algorithm3_data - algorithm1_data
    pairs_order.append(filename)
    for i in range(len(TALD_gain)):
        if  method_names[i] not in list(TALD_gain_dict.keys()):
            TALD_gain_dict[method_names[i]] = np.empty((0,))
            CKD_gain_dict[method_names[i]] = np.empty((0,))
        TALD_gain_dict[method_names[i]] = np.append(TALD_gain_dict[method_names[i]], np.array([TALD_gain[i]]))
        CKD_gain_dict[method_names[i]] = np.append(CKD_gain_dict[method_names[i]], np.array([coded_gain[i]]))

def main(method_names, data, filename, title, legend_flag=False):

    algorithm1_data, algorithm2_data, algorithm3_data = data

    # Create a numpy array of bar positions
    bar_positions = np.arange(len(method_names))
    # Set the seaborn style
    # sns.set_style('whitegrid')
    # sns.set()
    
    # # Width of each bar
    # bar_width = 0.2
    # plt.rcParams["figure.figsize"] = (figure_width_inches, figure_height_inches)
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = label_size
    # plt.rcParams['xtick.labelsize'] = ticks_size
    # plt.rcParams['ytick.labelsize'] = ticks_size



    # Define a color palette
    # color_palette = sns.color_palette("viridis")
    color_palette = sns.color_palette("rocket", n_colors=3)



    # Plotting the bars with color palette
    # plt.bar(bar_positions + 2*bar_width, algorithm3_data, bar_width, color=color_palette[2], label='Coded Teacher')
    # plt.bar(bar_positions + bar_width, algorithm2_data, bar_width, color=color_palette[1], label='TALD')
    # plt.bar(bar_positions, algorithm1_data, bar_width, color=color_palette[0], label='Underlying Method')
    # legend_txt = [r'$\bf Coded~Teacher$'+' \nexploited in\nunderlying method',
    #               r'$\bf TALD$'+ '\nexploited in\nunderlying method',
    #               'Underlying Method'
    #               ]
    
    legend_txt = ['Coded Teacher\non top of the\nunderlying method',
                  'TALD\non top of the\nunderlying method',
                  'Underlying Method'
                  ]
    # legend_txt = ['Coded Teacher\nexploited in\nunderlying method',
    #             'TALD\nexploited in\nunderlying method',
    #             'Underlying Method'
    #             ]``
    plt.bar(bar_positions + 2*bar_width, algorithm3_data, bar_width, color=color_palette[2], label=legend_txt[0])
    plt.bar(bar_positions + bar_width, algorithm2_data, bar_width, color=color_palette[1], label=legend_txt[1])
    plt.bar(bar_positions, algorithm1_data, bar_width, color=color_palette[0], label=legend_txt[2])
    
    

    plt.ylim(min(algorithm1_data) - 1, max(algorithm2_data) + 0.1)

    # # Adding labels and title
    # plt.xlabel('Methods')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    # plt.title(filename)

    # Adding x-axis ticks and labels
    plt.xticks(bar_positions + bar_width, method_names)

    # Adding legend
    legend = plt.legend(bbox_to_anchor=(1, 0.8), loc='best')
    # get the width of your widest label, since every label will need 
    # to shift by this amount after we align to the right
    # shift = max([t.get_window_extent().width for t in legend.get_texts()]) 
    shift = 1
    for t in legend.get_texts():
        t.set_ha('left') # ha is alias for horizontalalignment
        t.set_position((shift,0))
    # plt.legend(bbox_to_anchor=(0.5, h_space), loc='upper center', ncol=3)
    # Add legend with custom font properties
    # if legend_flag:   
    # plt.legend(prop={'family': 'Times New Roman', 'size': label_size},bbox_to_anchor=(0.5, h_space), loc='upper center', ncol=3)

    # Displaying the plot
    # breakpoint()
    plt.savefig(filename+".png", dpi=600, bbox_inches='tight')
    plt.savefig(filename+".pdf", dpi=600, bbox_inches='tight')
    plt.figure()


def main_KD(method_names, data, filename, title, legend_flag=False):

    algorithm1_data, algorithm2_data, algorithm3_data, algorithm4_data = data

    # Create a numpy array of bar positions
    bar_positions = np.arange(len(method_names))


    # Set the seaborn style
    sns.set_style('whitegrid')
    
    
    # Width of each bar
    bar_width = 0.2
    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = label_size
    plt.rcParams['xtick.labelsize'] = label_size

    # Define a color palette
    # color_palette = sns.color_palette("viridis")
    color_palette = sns.color_palette("rocket", n_colors=4)


    # Plotting the bars with color palette
    plt.bar(bar_positions + 3*bar_width, algorithm4_data, bar_width, color=color_palette[3], label='\textit{coded} Teacher')
    plt.bar(bar_positions + 2*bar_width, algorithm3_data, bar_width, color=color_palette[2], label='TALD')
    plt.bar(bar_positions + bar_width, algorithm2_data, bar_width, color=color_palette[1], label='BSS')
    plt.bar(bar_positions, algorithm1_data, bar_width, color=color_palette[0], label='KD')


    plt.ylim(min(algorithm1_data) - 1, max(algorithm4_data) + 0.5)

    # Adding x-axis ticks and labels
    plt.xticks(bar_positions + bar_width, method_names, rotation = 45)

    plt.legend(prop={'family': 'Times New Roman', 'size': 20},bbox_to_anchor=(0.5, h_space), loc='upper center', ncol=4)

    # Displaying the plot
    plt.savefig(filename+".png", dpi=600, bbox_inches='tight')
    plt.savefig(filename+".pdf", dpi=600, bbox_inches='tight')
    plt.figure()




if __name__ == "__main__":
    # Data for three algorithms for each method
    # Method names on the x-axis
    filename = "vgg13__MobileNetV2"
    title =  "T:vgg13==>S:MobileNetV2"
    
    method_names = ['FT', 'FitNet', 'CC', 'SP', 'RKD']

    algorithm1_data = [61.78, 64.14 , 64.86, 66.30, 64.52] # Underlying Method 
    algorithm2_data = [62.40, 64.49 , 65.90, 66.60, 65.35]
    algorithm3_data = [66.39, 65.79 , 65.83, 65.73, 65.88]

    data = [ 
                algorithm1_data, 
                algorithm2_data,
                algorithm3_data
            ]
    
    analysis(method_names, data, filename, title)
    main(method_names, data, filename, title)


    filename = "wrn_40_2__wrn16_2"
    title = "T:wrn_40_2==>S:wrn16_2" 
    method_names = ['FT', 'FitNet', 'CC', 'SP', 'RKD']

    algorithm1_data = [73.25, 73.58 , 73.56, 73.83, 73.35] # Underlying Method 
    algorithm2_data = [73.3 , 73.95 , 73.65, 74.08 , 73.70]
    algorithm3_data = [74.04, 74    , 73.91, 74.42, 73.64]

    data = [ 
                algorithm1_data, 
                algorithm2_data,
                algorithm3_data
            ]
    
    analysis(method_names, data, filename, title)

    # main(method_names, data, filename, title)


    filename = "wrn_40_2__wrn40_1"
    title = "T:wrn_40_2==>S:wrn40_1" 
    method_names = ['FT', 'FitNet', 'CC', 'SP', 'RKD']

    algorithm1_data = [71.59 , 72.24 , 72.21 , 72.43, 72.22] # Underlying Method 
    algorithm2_data = [71.65 , 72.45 , 72.8  , 72.5, 71.9 ]
    algorithm3_data = [72.31 , 72.83 , 72.32 , 72.87, 72.19]

    data = [ 
                algorithm1_data, 
                algorithm2_data,
                algorithm3_data
            ]
    analysis(method_names, data, filename, title)
    # main(method_names, data, filename, title)


    filename = "resnet32x4__resnet8x4"
    title =  "T:resnet32x4==>S:resnet8x4"
    
    method_names = ['FT', 'FitNet', 'CC', 'SP', 'RKD']

    algorithm1_data = [72.86, 73.50 , 72.97, 72.94, 71.90] # Underlying Method 
    algorithm2_data = [73.18, 73.79 , 73.35, 73.82, 72.6]
    algorithm3_data = [73.15, 73.91 , 73.16, 73.17, 72.95]

    data = [ 
                algorithm1_data, 
                algorithm2_data,
                algorithm3_data
            ]
    analysis(method_names, data, filename, title)
    # main(method_names, data, filename, title)





    filename = "resnet50__MobileNetV2"
    title =  "T:resnet50==>S:MobileNetV2"
    
    method_names = ['FT', 'FitNet', 'CC', 'SP', 'RKD']

    algorithm1_data = [60.99, 63.16 , 65.43, 68.08, 64.43] # Underlying Method 
    algorithm2_data = [61.40, 63.96 , 65.80, 67.25, 64.90]
    algorithm3_data = [66.34, 66.08 , 66.00, 65.91, 65.42]

    data = [ 
                algorithm1_data, 
                algorithm2_data,
                algorithm3_data
            ]
    analysis(method_names, data, filename, title)
    main(method_names, data, filename, title)


    for idx, key in enumerate(method_names):
        print(pairs_order[idx], key)
        print( CKD_gain_dict[key], TALD_gain_dict[key] )
        print( CKD_gain_dict[key] - TALD_gain_dict[key] )

# if __name__ == "__main__":
#     # KD ONLY CKD ... BSS ... TALD 

#     filename = "resnet32x4__resnet8x4_KD"
#     title =  "T:resnet32x4-->S:resnet8x4"
    
#     method_names = ["T:resnet32x4->S:resnet8x4",]

#     algorithm1_data = [73.33] # Underlying Method 
#     algorithm2_data = [73.53]
#     algorithm3_data = [73.73]
#     algorithm4_data = [74.41]

#     data = [ 
#                 algorithm1_data, 
#                 algorithm2_data,
#                 algorithm3_data,
#                 algorithm4_data
#             ]
    
#     main_KD(method_names, data, filename, title)