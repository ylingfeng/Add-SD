prompt_edit_add = """
You are an AI assistant tasked with generating edit instructions based on the given samples. Here are some requests:

1. The instruction can be a question or an imperative sentence. The response should be short and concise, ideally one phrase or one sentence long in English.

2. The instruction is used to ask or command to add an instance of a single category to a real image. The added category is denoted as <obj>, for example: add a <obj> to the image. The symbol <obj> must be in the instruction, and only one <obj> symbol should be contained.

3. The language of instructions and responses should be diverse. Try to use different verbs for each instruction and each response to maximize diversity. Either an imperative sentence or a question is permitted.

Here are some examples:
add a <obj>
generate a <obj> on the figure
"""


# results_chatgpt
prompt_edit_add = [
    "add a <obj>",
    "Incorporate a <obj> into the picture.",
    "Can you add a <obj> to the image?",
    "Insert a <obj> onto the scene.",
    "Could you incorporate a <obj> in the frame?",
    "Place a <obj> into the picture.",
    "Is it possible to generate a <obj> on the image?",
    "Integrate a <obj> into the composition.",
    "Would you mind adding a <obj> to the scene?",
    "Embed a <obj> into the visual.",
    "How about including a <obj> in the image?",

    "Add a <obj> to the photo, please.",
    "Create a <obj> within the image.",
    "Place a <obj> onto the existing picture.",
    "Would you consider inserting a <obj> into the composition?",
    "Formulate a <obj> on the visual.",
    "Integrate a <obj> into the scene for enhancement.",
    "Can you add a <obj> to the current image?",
    "Generate a <obj> to enrich the picture.",
    "Embed a <obj> into the frame.",
    "Consider adding a <obj> to the image.",

    "Add a <obj>.",
    "Put in a <obj>?",
    "Include a <obj>.",
    "Insert a <obj>.",
    "Place a <obj>.",
    "Generate a <obj>?",
    "Integrate a <obj>.",
    "Add a <obj> here.",
    "Drop a <obj>.",
    "How about a <obj>?",


    "Attach a <obj>.",
    "Include a <obj> in it.",
    "Place a <obj> in the picture.",
    "Pop a <obj> in there.",
    "Snap in a <obj>.",
    "Plant a <obj> in the scene.",
    "Slide a <obj> in.",
    "Toss a <obj> in the mix.",
    "Blend a <obj> in.",
    "Inject a <obj>.",
]
