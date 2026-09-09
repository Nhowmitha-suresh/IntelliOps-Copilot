import os

def resolve_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_from_base = os.path.join(base_dir, rel_path)
    if os.path.exists(os.path.dirname(path_from_base)):
        return path_from_base
    path_with_prefix = os.path.join("intelliops-copilot", rel_path)
    return path_with_prefix

SFT_PAIRS = [
    # Monologues
    ("write a short monologue about betrayal", "O treacherous friend! Thou hast pierced my heart with subtle words. Was ever love repaid with such deceit? The heavens weep for my lost faith."),
    ("write a short monologue about courage", "Stand firm, brave hearts! Though the storm rages and night covers the battlefield, victory favors the valiant who fear not death."),
    ("write a monologue about ambition", "Ambition burns like a fever in my blood. A crown beckons from afar, and no moral boundary shall stay my march to destiny."),
    ("write a monologue about sorrow", "Grief weighs upon my soul like lead. The sun has lost its radiance, and memories of joy serve only to deepen this darkness."),
    ("write a monologue about honor", "Honor is the true gold of a noble mind. Without it, a prince is but a beggar wrapped in silk, naked before truth."),
    ("write a monologue about revenge", "Revenge shall be my shadow until the score is settled. Blood demands blood, and I shall not sleep till justice is done."),
    ("write a monologue about love", "My love for thee outshines the morning star. Though distance separates our hands, our souls remain forever bound as one."),
    ("write a monologue about fate", "We are but chess pieces upon the board of destiny. Stars shape our rise and fall, yet we fight against the thread of fate."),
    ("write a monologue about night", "Dark night descends upon the castle walls, cloaking secrets in black velvet. Spirits whisper in the chill wind."),
    ("write a monologue about time", "Time moves with swift and silent steps. It steals youth from beauty and turns mighty empires to dust."),

    # Scene Continuations
    ("continue this scene: KING:", "KING: Speak, noble lord! What news from the bloody field?\nWARWICK: Victory is ours, my liege! The enemy turns in retreat!"),
    ("continue this scene: MACBETH:", "MACBETH: Is this a dagger which I see before me?\nLADY MACBETH: A foolish thought, to say a sorry sight. Be steadfast in our purpose!"),
    ("continue this scene: HAMLET:", "HAMLET: To be, or not to be, that is the question.\nHORATIO: My lord, sweet thoughts should temper this dark mood."),
    ("continue this scene: ROMEO:", "ROMEO: Soft! What light through yonder window breaks?\nJULIET: O Romeo, Romeo! Wherefore art thou Romeo?"),
    ("continue this scene: OTHELLO:", "OTHELLO: Give me the ocular proof of her dishonor!\nIAGO: Beware, my lord, of jealousy; it is the green-eyed monster."),
    ("continue this scene: LEAR:", "LEAR: Blow, winds, and crack your cheeks! Rage! Blow!\nFOOL: O uncle, court holy-water in a dry house is better than this rain-water out o' door."),
    ("continue this scene: BRUTUS:", "BRUTUS: Into what dangers would you lead me, Cassius?\nCASSIUS: To set yourself upright and free Rome from tyranny!"),
    ("continue this scene: PROSPERO:", "PROSPERO: My potent art is ended; now my charms are all o'erthrown.\nARIEL: Master, your freedom awaits upon the tempest."),
    ("continue this scene: ROSALIND:", "ROSALIND: Love is merely a madness, and deserves as well a dark house and a whip.\nORLANDO: Did you ever hear of any that died for love?"),
    ("continue this scene: BEATRICE:", "BEATRICE: I would rather hear my dog bark at a crow than a man swear he loves me.\nBENEDICK: God keep your ladyship still in that mind!"),

    # Dialogues
    ("write a dialogue between two characters about ambition", "CASSIUS: Why should Caesar dead set the world in awe?\nBRUTUS: He would be crowned; how that might change his nature, there's the question."),
    ("write a dialogue about loyalty", "KENT: Royal Lear, whom I have ever honor'd as my king.\nLEAR: Out of my sight! Peace, Kent! Come not between the dragon and his wrath."),
    ("write a dialogue about jealousy", "IAGO: Look to your wife; observe her well with Cassio.\nOTHELLO: Dost thou say so? I'll tear her all to pieces!"),
    ("write a dialogue about war", "HENRY: Once more unto the breach, dear friends, once more!\nEXETER: The game's afoot: follow your spirit, and upon this charge cry God for Harry!"),
    ("write a dialogue about justice", "PORTIA: The quality of mercy is not strained; it droppeth as the gentle rain from heaven.\nSHYLOCK: My deeds upon my head! I crave the law!"),
    ("write a dialogue about grief", "MACDUFF: All my pretty ones? Did you say all? O hell-kite! All?\nMALCOLM: Dispute it like a man.\nMACDUFF: I shall do so; but I must also feel it as a man."),
    ("write a dialogue about power", "RICHARD: Ay, hand from head: for so I will, or die.\nBUCKINGHAM: My lord, the people crowd to crown your brow."),
    ("write a dialogue about truth", "POLONIUS: To thine own self be true, and it must follow, as the night the day, thou canst not then be false to any man."),
    ("write a dialogue about fate", "ROMEO: I defy you, stars!\nBALTHASAR: I do beseech you, sir, have patience: your looks are pale and wild."),
    ("write a dialogue about winter", "GLOUCESTER: Now is the winter of our discontent made glorious summer by this sun of York."),

    # Character Speeches & Advice
    ("write a speech from a king to his army", "Brave soldiers, heroes of the realm! This day we fight not for gold, but for the liberty of our children and the honor of our crown. Forward to victory!"),
    ("write a speech from a villain planning revenge", "They call me villain, yet they taught me cruelty. I shall smile and smile while plotting their downfall. None shall escape my trap."),
    ("write a speech from a fool offering wisdom", "A wise man knows himself to be a fool, but a fool thinks himself wise. Better a witty fool than a foolish wit, my noble lord!"),
    ("write advice from an old sage", "Hearken to me, young prince: let temperance guide thy sword and wisdom temper thy rage. True strength lies in self-command."),
    ("write a queen's address to her court", "My lords and ladies, the crown demands unity in this hour of peril. Let no petty feud divide our council while danger knocks upon our gates."),
    ("write a soldier's farewell before battle", "Farewell, sweet lady. If I fall upon the bloody field, know that thy name was the last breath upon my lips."),
    ("write a ghost's warning to a prince", "Mark me well, young prince! Revenge my foul and most unnatural murder. Swear by my sword that thou wilt remember me!"),
    ("write a lover's oath under the stars", "By yonder blessed moon I swear, my heart is thine through life and death. No storm shall shake my devotion."),
    ("write a scholar's meditation on nature", "Nature speaks in silence to those who listen. In every leaf and stream lies a secret testament of divine craft."),
    ("write a knight's oath of chivalry", "I swear to defend the weak, uphold the truth, and serve the king with loyal heart until my final breath."),

    # Short Prompts & Soliloquies
    ("write a soliloquy on conscience", "Conscience doth make cowards of us all. The heavy burden of guilt turns resolution into pale hesitation."),
    ("write a soliloquy on sleep", "O gentle sleep, nature's soft nurse, how have I frighted thee, that thou no more wilt weigh my eyelids down?"),
    ("write a soliloquy on vanity", "What is human glory but a passing shadow? We build monuments of stone, yet time erases every name."),
    ("write a soliloquy on hope", "Hope is a flickering candle in the dark. Though winds threaten to blow it out, its light keeps despair at bay."),
    ("write a soliloquy on guilt", "Will all great Neptune's ocean wash this blood clean from my hand? No, this my hand will rather the multitudinous seas incarnadine."),
    ("write a short poem on spring", "When daisies pied and violets blue, and lady-smocks all silver-white, do paint the meadows with delight."),
    ("write a short poem on mortality", "Golden lads and girls all must, as chimney-sweepers, come to dust."),
    ("write a short line about friendship", "A faithful friend is a strong defense, and he that hath found such a one hath found a treasure."),
    ("write a warning against pride", "Pride goeth before destruction, and a haughty spirit before a fall. Humility is the shield of the righteous."),
    ("write a plea for peace", "Lay down your swords, good lords! Let not brother shed brother's blood in this unnatural strife.")
]

# Multiply patterns with slight variations to reach ~150 training samples
def generate_sft_dataset(output_path: str = "data/processed/sft_data.txt") -> str:
    full_path = resolve_path(output_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    formatted_items = []
    # Repeat the dataset 3 times to ensure sufficient iterations for epoch training
    for _ in range(3):
        for inst, resp in SFT_PAIRS:
            text = f"### Instruction:\n{inst}\n### Response:\n{resp}\n"
            formatted_items.append(text)
            
    dataset_content = "\n".join(formatted_items)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(dataset_content)
        
    print(f"Generated SFT dataset with {len(formatted_items)} samples ({len(dataset_content):,} characters) at: {full_path}")
    return full_path

if __name__ == "__main__":
    generate_sft_dataset()
